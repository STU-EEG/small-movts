import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

from src.model import VideoDataset, Simple3DCNN, UnifiedBlinkDetector, collate_fn, validate

def compute_all_metrics(model, dataloader, device): 
    """
    Compute comprehensive classification metrics.
    Returns: dict with all metrics, plus predictions and labels for detailed analysis
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    
    # Per-class metrics (class 0 = non-blink, class 1 = blink)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0) * 100
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0) * 100
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0) * 100
    return metrics

def main():
    config = {
        'batch_size': 4,
        'max_frames': 30,
        'num_workers': 2,
        'checkpoint_dir': 'checkpoints',
        'test_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/b',
        'test_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/nb',
        'test_subject2_b': '/share/Share/valera/veeg_july19_2025_vkuklin/videos_cropped/accepted/b',
        'test_subject2_nb': '/share/Share/valera/veeg_july19_2025_vkuklin/videos_cropped/accepted/nb'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load test data
    test_dataset = torch.utils.data.ConcatDataset([
        VideoDataset(config['test_b_dir'], label=1, max_frames=config['max_frames']),
        VideoDataset(config['test_nb_dir'], label=0, max_frames=config['max_frames'])
    ])
    test_subject2_dataset = torch.utils.data.ConcatDataset([
        VideoDataset(config['test_subject2_b'], label=1, max_frames=config['max_frames']),
        VideoDataset(config['test_subject2_nb'], label=0, max_frames=config['max_frames'])
    ])
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'], 
                           collate_fn=collate_fn)
    
    test_subject2_loader = DataLoader(
        test_subject2_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'], 
        collate_fn=collate_fn
    )
    
    
    criterion = nn.CrossEntropyLoss()
    arch_dirs = [p for p in Path(config['checkpoint_dir']).iterdir() if p.is_dir()]

    results = {}
    for arch_dir in sorted(arch_dirs):
        arch_name = arch_dir.name
        checkpoint_path = arch_dir / 'best_checkpoint.pth'
    
        if not checkpoint_path.exists():
            print(f"Skipping {arch_name}: best_checkpoint.pth not found")
            continue
        
        try:
            print(f"Checkpoint: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model = UnifiedBlinkDetector(
                architecture=arch_name, 
                num_classes=2, 
                input_channels=3
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint (val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
            
            test_loss, test_acc = validate(model, test_loader, criterion, device, verbose=False)
            metrics = compute_all_metrics(model, test_loader, device)
            
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%")

            test_loss2, test_acc2 = validate(model, test_subject2_loader, criterion, device, verbose=False)
            metrics_subject2 = compute_all_metrics(model, test_subject2_loader, device)
            print(f"\n unseen subject - generalization test:")
            print(f"   Test Loss: {test_loss2:.4f}")
            print(f"   Test Accuracy: {test_acc2:.2f}%")
            print("="*60)
            
            
            results[arch_name] = {
                'metrics': metrics,
                'metrics_subj2': metrics_subject2,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'val_acc': checkpoint.get('val_acc', 0),
                'subject2_acc': test_acc2,
                'subject2_loss': test_loss2,
                'epoch': checkpoint.get('epoch', 'N/A')
            }
            
        except Exception as e:
            print(f"Error evaluating {arch_name}: {str(e)}")
            results[arch_name] = {'error': str(e)}
    
    print(results)


if __name__ == "__main__":
    main()