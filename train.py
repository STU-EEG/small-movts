import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

from src.model import VideoDataset, Simple3DCNN, UnifiedBlinkDetector, collate_fn, train_epoch, validate


def main():
    # Configuration
    config = {
        'batch_size': 4,  # Small batch size due to 3D videos
        'num_epochs': 50,
        'learning_rate': 0.001,
        'max_frames': 30,  # Approximately 1s at 30fps
        'num_workers': 2,
        'checkpoint_dir': 'checkpoints_repetition1',
        'train_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/train/b',
        'train_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/train/nb',
        'test_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/b',
        'test_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/nb'
    }
    
    architectures = ['basic', 'resnet', 'resnet_attention', 'multiscale', 'twostream', 'advanced']

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

    results = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training architecture: {arch}")
        print('='*60)
        
        # Create architecture-specific checkpoint directory
        arch_checkpoint_dir = os.path.join(config['checkpoint_dir'], arch)
        os.makedirs(arch_checkpoint_dir, exist_ok=True)
        
        try:
            # Initialize model with current architecture
            model = UnifiedBlinkDetector(architecture=arch, num_classes=2, input_channels=3).to(device)
            model_info = model.get_architecture_info()
            print(f"Parameters: {model_info['total_parameters']:,}")
            print(f"Model size: {model_info['model_size_mb']:.2f} MB")

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
                    'architecture': arch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config
                }
                
                # Save latest checkpoint
                torch.save(checkpoint, os.path.join(arch_checkpoint_dir, 'latest_checkpoint.pth'))
                
                # Save best checkpoint
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(checkpoint, os.path.join(arch_checkpoint_dir, 'best_checkpoint.pth'))
                    print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            print(f'\nCompleted {arch}! Best validation accuracy: {best_val_acc:.2f}%')
            results[arch] = {
                'best_val_acc': best_val_acc,
                'status': 'completed',
                'checkpoint_dir': arch_checkpoint_dir
            }
            
        except Exception as e:
            print(f"\nERROR with architecture {arch}: {str(e)}")
            results[arch] = {
                'error': str(e),
                'status': 'failed'
            }
            continue
    
    # Save results summary
    results_path = os.path.join(config['checkpoint_dir'], 'architecture_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print('='*60)
    for arch, result in results.items():
        if result['status'] == 'completed':
            print(f"{arch}: Best Val Acc = {result['best_val_acc']:.2f}%")
        else:
            print(f"{arch}: FAILED - {result.get('error', 'Unknown error')}")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
