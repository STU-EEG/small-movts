from collections import defaultdict
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

from src.model import VideoDataset, PartialVideoDataset, UnifiedBlinkDetector, collate_fn, validate, train_test_split, get_blink_nonblink_video_paths

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
        'checkpoint_dir': 'checkpoints_multiTrial_sep2026',
        'subj1_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted',
        'test_subject2_b': '/share/Share/valera/veeg_july19_2025_vkuklin/videos_cropped/accepted/b',
        'test_subject2_nb': '/share/Share/valera/veeg_july19_2025_vkuklin/videos_cropped/accepted/nb',
        'out_filename': 'metrics.json',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    subj1_dir = Path(config['subj1_dir'])
    paths_b, paths_nb = get_blink_nonblink_video_paths(subj1_dir)
    
    # Load test data
    test_subject2_dataset = torch.utils.data.ConcatDataset([
        VideoDataset(config['test_subject2_b'], label=1, max_frames=config['max_frames']),
        VideoDataset(config['test_subject2_nb'], label=0, max_frames=config['max_frames'])
    ])
    test_subject2_loader = DataLoader(
            test_subject2_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'], 
            collate_fn=collate_fn
        )
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(config['checkpoint_dir'])
    arch_dirs = [p for p in checkpoint_dir.iterdir() if p.is_dir()]

    
    results: dict[str, list[dict]] = defaultdict(list)
    for train_split_seed in range(20):
        _, paths_b_test = train_test_split(paths_b, train_split_seed)
        _, paths_nb_test = train_test_split(paths_nb, train_split_seed)
        test_dataset = torch.utils.data.ConcatDataset([
                PartialVideoDataset(paths_b_test, label=1, max_frames=config['max_frames']),
                PartialVideoDataset(paths_nb_test, label=0, max_frames=config['max_frames'])
            ])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'], 
                            collate_fn=collate_fn)


        for arch_dir in sorted(arch_dirs):
            arch_name = arch_dir.name
            checkp_file_name = f'seed{train_split_seed}_best_checkpoint.pth'
            checkpoint_path = arch_dir / checkp_file_name
        
            if not checkpoint_path.exists():
                print(f"Skipping {arch_name}: {checkp_file_name} not found")
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
                
                
                results[arch_name].append({
                    'metrics': metrics,
                    'metrics_subj2': metrics_subject2,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'val_acc': checkpoint.get('val_acc', 0),
                    'subject2_acc': test_acc2,
                    'subject2_loss': test_loss2,
                    'epoch': checkpoint.get('epoch', 'N/A')
                })
                
            except Exception as e:
                print(f"Error evaluating {arch_name}: {str(e)}")
                results[arch_name].append({'error': str(e)})
        
    print(results)
    out_path = checkpoint_dir / config['out_filename']
    with out_path.open('w') as f:
        json.dump(results, f, indent=4)
    print(f"saved results to {out_path}")



if __name__ == "__main__":
    main()