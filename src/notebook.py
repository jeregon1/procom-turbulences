import argparse
import glob
import torch
from turbulence.dataset import load_data
from turbulence.train import train

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description="Train turbulence autoencoder with spectral loss")
    parser.add_argument('--nb_images', type=int, default=500, help='Number of images to load')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set fraction')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1250, help='Number of training epochs')
    parser.add_argument('--b', type=float, default=0.2, help='Spectral loss weight')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()

    # Data loading
    train_loader, val_loader, test_loader = load_data(
        './velocity_images', args.nb_images, args.val_size, args.test_size, args.batch_size
    )

    # Set model name
    name = f"spectralLoss_b{args.b}"
    pretrained = False
    model_path = args.pretrained_model_path
    if args.resume:
        # find latest checkpoint for this run
        ckpt = glob.glob(f"./turbulence/pretrained/{name}_latest.ckpt")
        if ckpt:
            model_path = ckpt[0]
            pretrained = True
    elif args.pretrained_model_path:
        pretrained = True

    train(
        model_path or '',
        train_loader,
        val_loader,
        epochs=args.epochs,
        pretrained=pretrained,
        save_name=name,
        spectralB=args.b
    )

if __name__ == '__main__':
    main()
