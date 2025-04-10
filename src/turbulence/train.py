import os
import torch
import pytorch_lightning as pl
from turbulence.network import Turbulence
from turbulence.utils import plot_losses

def save_checkpoint(model, epoch, model_path):
    print("Saving model checkpoint...")
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, model_path)
    print(f"Model checkpoint saved.")

def train(model_path, train_loader, val_loader, epochs=1000, pretrained=False):
    model = Turbulence()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0
    if pretrained:
        print("Loading pre-trained model...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}.")
    else : 
        print("No pre-trained model found. Training from scratch.")

    trainer = pl.Trainer(max_epochs=epochs)
    model.train()
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")
    
    # Saving the checkpoint after training
    saving_path = f'./turbulence/pretrained/turbulence_epoch_{start_epoch + epochs}.ckpt'
    save_checkpoint(model, start_epoch + epochs, saving_path)

    # Plotting the training and validation losses
    plot_losses(model.TRAINING_LOSSES, model.VALIDATION_LOSSES)
