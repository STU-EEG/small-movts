import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import VideoDataset, Simple3DCNN, collate_fn, validate


def main():
    config = {
        'batch_size': 4,
        'max_frames': 30,
        'num_workers': 2,
        'checkpoint_path': 'checkpoints/best_checkpoint.pth',
        'test_b_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/b',
        'test_nb_dir': '/share/Share/valera/veeg_june19_2025_vzuev/videos_cropped/accepted/test/nb'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model = Simple3DCNN(num_classes=2, input_channels=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (val_acc: {checkpoint.get('val_acc', 'N/A')}%)")
    
    # Load test data
    test_dataset = torch.utils.data.ConcatDataset([
        VideoDataset(config['test_b_dir'], label=1, max_frames=config['max_frames']),
        VideoDataset(config['test_nb_dir'], label=0, max_frames=config['max_frames'])
    ])
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'], 
                           collate_fn=collate_fn)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()