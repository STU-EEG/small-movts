import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import os
import glob
import cv2
import numpy as np
from pathlib import Path
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
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validation'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    # Configuration
    config = {
        'batch_size': 4,  # Small batch size due to 3D videos
        'num_epochs': 50,
        'learning_rate': 0.001,
        'max_frames': 30,  # Approximately 1s at 30fps
        'num_workers': 2,
        'checkpoint_dir': 'checkpoints',
        'train_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/train/b',
        'train_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/train/nb',
        'test_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/b',
        'test_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/nb'
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print("Loading datasets...")
    train_b_dataset = VideoDataset(config['train_b_dir'], label=1, max_frames=config['max_frames'])
    train_nb_dataset = VideoDataset(config['train_nb_dir'], label=0, max_frames=config['max_frames'])
    test_b_dataset = VideoDataset(config['test_b_dir'], label=1, max_frames=config['max_frames'])
    test_nb_dataset = VideoDataset(config['test_nb_dir'], label=0, max_frames=config['max_frames'])
    
    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset([train_b_dataset, train_nb_dataset])
    test_dataset = torch.utils.data.ConcatDataset([test_b_dataset, test_nb_dataset])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = Simple3DCNN(num_classes=2, input_channels=3).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 'best_checkpoint.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    main()
