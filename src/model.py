from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models.video as video_models
import os
import glob
import cv2
import numpy as np
import random
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Custom Dataset for video files with variable dimensions
class VideoDataset(Dataset):
    def __init__(self, data_dir, label, max_frames=30):
        """
        Args:
            data_dir: directory containing video files
            label: label for this class (0 for nb, 1 for b)
            max_frames: maximum number of frames to sample (approx 1s at 30fps)
        """
        self.data_dir = data_dir
        self.label = label
        self.max_frames = max_frames
        self.video_files = glob.glob(os.path.join(data_dir, '*'))
        
    def __len__(self):
        return len(self.video_files)
    
    def load_video(self, video_path):
        """Load video with variable dimensions"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Sample frames uniformly to max_frames
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert to tensor and normalize to [0, 1]
        frames = np.array(frames) / 255.0
        # Convert to (C, T, H, W) format
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video = self.load_video(video_path)
        
        if video is None:
            # Return a dummy tensor if video loading fails
            video = torch.zeros(3, self.max_frames, 224, 224)
        
        return video, self.label

class PartialVideoDataset(VideoDataset):
    def __init__(self, video_files: Sequence, label: int, max_frames=30):
        self.video_files = video_files
        self.label = label
        self.max_frames = max_frames

def get_blink_nonblink_video_paths(train_dir: Path) -> tuple[list[Path], list[Path]]:
    paths: list[Path] = list(train_dir.glob("*.mp4"))
    paths_b = [p for p in paths if p.stem.startswith('b')]
    paths_nb = [p for p in paths if p.stem.startswith('n')]
    print(f'found {len(paths_b)} blink and {len(paths_nb)} non-blink files in train dir {train_dir}')
    return paths_b, paths_nb
    

def train_test_split(
        arr: Sequence,
        seed: int,
        train_fraction: float = .6) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    arr_shuffled = np.random.permutation(arr)
    split_idx = int(train_fraction * len(arr_shuffled))
    return arr_shuffled[:split_idx], arr_shuffled[split_idx:]

# Collate function to handle variable dimensions
def collate_fn(batch):
    """Custom collate function for variable-sized videos"""
    videos, labels = zip(*batch)
    
    # Find max height and width in batch
    max_h = max(video.shape[2] for video in videos)
    max_w = max(video.shape[3] for video in videos)
    max_t = max(video.shape[1] for video in videos)
    
    # Pad videos to same dimensions
    padded_videos = []
    for video in videos:
        c, t, h, w = video.shape
        pad_t = max_t - t
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad (T, H, W)
        padded = F.pad(video, (0, pad_w, 0, pad_h, 0, pad_t))
        padded_videos.append(padded)
    
    return torch.stack(padded_videos), torch.LongTensor(labels)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                                padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection (1x1 conv if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        batch, channels, t, h, w = x.size()
        # Squeeze
        y = x.view(batch, channels, -1).mean(dim=2)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1, 1)
        return x * y


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module for 3D"""
    def __init__(self, channels, reduction=16, spatial_kernel=3):
        super().__init__()
        self.se = SEBlock3D(channels, reduction)
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel, 
                              padding=spatial_kernel//2)
        
    def forward(self, x):
        x = self.se(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.conv(spatial_input))
        return x * spatial_attention


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels//4, 
                                 kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.branch2 = nn.Conv3d(in_channels, out_channels//4, 
                                 kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.branch3 = nn.Conv3d(in_channels, out_channels//4, 
                                 kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.branch4 = nn.Conv3d(in_channels, out_channels//4, 
                                 kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return F.relu(self.bn(out))

# 3D CNN Model
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super(Simple3DCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class UnifiedBlinkDetector(nn.Module):
    """
    Unified 3D CNN architecture for blink detection with multiple options.
    
    Args:
        architecture (str): One of the following options:
            - 'basic': Original 3-layer 3D CNN
            - 'resnet': 3D CNN with residual connections
            - 'resnet_attention': ResNet + CBAM attention
            - 'multiscale': Multi-scale temporal processing
            - 'twostream': Two-stream (RGB + motion) architecture
            - 'pretrained': Pre-trained R3D-18 from torchvision
            - 'advanced': Full advanced architecture (residuals + attention + temporal attention)
        
        num_classes (int): Number of output classes (default: 2)
        input_channels (int): Number of input channels (default: 3)
        pretrained_backbone (bool): For 'pretrained' architecture, use pretrained weights
    """
    
    def __init__(self, architecture='advanced', num_classes=2, input_channels=3, 
                 pretrained_backbone=True):
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Select architecture
        if architecture == 'basic':
            self.model = self._build_basic(num_classes, input_channels)
        elif architecture == 'resnet':
            self.model = self._build_resnet(num_classes, input_channels)
        elif architecture == 'resnet_attention':
            self.model = self._build_resnet_attention(num_classes, input_channels)
        elif architecture == 'multiscale':
            self.model = self._build_multiscale(num_classes, input_channels)
        elif architecture == 'twostream':
            self.model = self._build_twostream(num_classes, input_channels)
        elif architecture == 'pretrained':
            self.model = self._build_pretrained(num_classes, input_channels, pretrained_backbone)
        elif architecture == 'advanced':
            self.model = self._build_advanced(num_classes, input_channels)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: 'basic', 'resnet', 'resnet_attention', "
                           f"'multiscale', 'twostream', 'pretrained', 'advanced'")
    
    def _build_basic(self, num_classes, input_channels):
        """Original simple 3-layer 3D CNN"""
        return nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_resnet_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _build_resnet(self, num_classes, input_channels):
        """3D CNN with residual connections"""
        class ResNet3D(nn.Module):
            def __init__(self, parent, num_classes, input_channels):
                super().__init__()
                self.parent = parent
                self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm3d(32)
                self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
                
                self.layer1 = parent._make_resnet_layer(32, 64, blocks=2, stride=2)
                self.layer2 = parent._make_resnet_layer(64, 128, blocks=2, stride=2)
                self.layer3 = parent._make_resnet_layer(128, 256, blocks=2, stride=2)
                
                self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc1 = nn.Linear(256, 512)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(512, num_classes)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return ResNet3D(self, num_classes, input_channels)
    
    def _build_resnet_attention(self, num_classes, input_channels):
        """ResNet with CBAM attention"""
        class ResNetAttention3D(nn.Module):
            def __init__(self, parent, num_classes, input_channels):
                super().__init__()
                self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm3d(32)
                self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
                
                self.layer1 = parent._make_resnet_layer(32, 64, blocks=2, stride=2)
                self.attn1 = CBAM3D(64)
                self.layer2 = parent._make_resnet_layer(64, 128, blocks=2, stride=2)
                self.attn2 = CBAM3D(128)
                self.layer3 = parent._make_resnet_layer(128, 256, blocks=2, stride=2)
                self.attn3 = CBAM3D(256)
                
                self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc1 = nn.Linear(256, 512)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(512, num_classes)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                x = self.layer1(x)
                x = self.attn1(x)
                x = self.layer2(x)
                x = self.attn2(x)
                x = self.layer3(x)
                x = self.attn3(x)
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return ResNetAttention3D(self, num_classes, input_channels)
    
    def _build_multiscale(self, num_classes, input_channels):
        """Multi-scale temporal processing"""
        class MultiScale3D(nn.Module):
            def __init__(self, num_classes, input_channels):
                super().__init__()
                self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm3d(32)
                self.pool1 = nn.MaxPool3d(kernel_size=2)
                
                self.multiscale = MultiScaleTemporalBlock(32, 64)
                self.pool2 = nn.MaxPool3d(kernel_size=2)
                
                self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm3d(128)
                self.pool3 = nn.MaxPool3d(kernel_size=2)
                
                self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
                self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                x = self.multiscale(x)
                x = self.pool2(x)
                x = F.relu(self.bn3(self.conv3(x)))
                x = self.pool3(x)
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return MultiScale3D(num_classes, input_channels)
    
    def _build_twostream(self, num_classes, input_channels):
        """Two-stream (RGB + motion) architecture"""
        class TwoStream3D(nn.Module):
            def __init__(self, num_classes, input_channels):
                super().__init__()
                # RGB stream (simplified resnet)
                self.rgb_stream = nn.Sequential(
                    nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2),
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten()
                )
                
                # Motion stream (same architecture)
                self.motion_stream = nn.Sequential(
                    nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2),
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten()
                )
                
                self.fusion = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                # Compute temporal differences
                motion = x[:, :, 1:] - x[:, :, :-1]
                motion = F.pad(motion, (0, 0, 0, 0, 0, 1))
                
                rgb_features = self.rgb_stream(x)
                motion_features = self.motion_stream(motion)
                
                combined = torch.cat([rgb_features, motion_features], dim=1)
                return self.fusion(combined)
        
        return TwoStream3D(num_classes, input_channels)
    
    def _build_pretrained(self, num_classes, input_channels, pretrained):
        """Pre-trained R3D-18 from torchvision"""
        class Pretrained3D(nn.Module):
            def __init__(self, num_classes, input_channels, pretrained):
                super().__init__()
                if input_channels != 3:
                    raise ValueError("Pretrained model expects 3 input channels")
                
                self.backbone = video_models.r3d_18(pretrained=pretrained)
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
                
                self.classifier = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        return Pretrained3D(num_classes, input_channels, pretrained)
    
    def _build_advanced(self, num_classes, input_channels):
        """Full advanced architecture with residuals, attention, and temporal attention"""
        class Advanced3D(nn.Module):
            def __init__(self, num_classes, input_channels):
                super().__init__()
                self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=5, stride=2, padding=2)
                self.bn1 = nn.BatchNorm3d(32)
                
                self.res1 = ResidualBlock3D(32, 64, stride=2)
                self.attn1 = CBAM3D(64)
                self.res2 = ResidualBlock3D(64, 128, stride=2)
                self.attn2 = CBAM3D(128)
                self.res3 = ResidualBlock3D(128, 256, stride=2)
                
                self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
                
                # Calculate feature dimension
                self.feature_dim = 256 * 4 * 4 * 4
                
                # Temporal attention
                self.temporal_attention = nn.MultiheadAttention(
                    embed_dim=self.feature_dim, num_heads=8, batch_first=True
                )
                self.layer_norm = nn.LayerNorm(self.feature_dim)
                
                self.classifier = nn.Sequential(
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.res1(x)
                x = self.attn1(x)
                x = self.res2(x)
                x = self.attn2(x)
                x = self.res3(x)
                x = self.adaptive_pool(x)
                
                # Reshape for temporal attention: (B, C*H*W, T)
                B, C, T, H, W = x.shape
                x = x.view(B, C, T * H * W).permute(0, 2, 1)  # (B, THW, C)
                
                # Apply temporal attention
                x = self.layer_norm(x)
                attended, _ = self.temporal_attention(x, x, x)
                x = attended.mean(dim=1)  # Global pooling
                
                return self.classifier(x)
        
        return Advanced3D(num_classes, input_channels)
    
    def forward(self, x):
        """Forward pass through the selected architecture"""
        return self.model(x)
    
    def get_architecture_info(self):
        """Return information about the current architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': self.architecture,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB
        }

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for videos, labels in tqdm(dataloader, desc='Training'):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

# Validation function
def validate(model, dataloader, criterion, device, verbose: bool = True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validation', disable = not verbose):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total
