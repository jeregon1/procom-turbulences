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

def train(model_path, train_loader, val_loader, epochs=1000, pretrained=False, save_name="turbulence", spectralB=1e-3):
    model = Turbulence(b=spectralB)
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

    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=5)
    model.train()
    try:
        trainer.fit(model, train_loader, val_loader)
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        saving_path = f'./turbulence/pretrained/{save_name}_interrupted_epoch_{trainer.current_epoch}.ckpt'
        save_checkpoint(model, start_epoch + trainer.current_epoch, saving_path)
        save_losses_to_csv(model.TRAINING_LOSSES, model.VALIDATION_LOSSES, spectralB, out_dir="/homes/j25lopez/pml/results/",model_name=save_name+"_interrupted")
        raise
    
    # Saving the checkpoint after training
    saving_path = f'./turbulence/pretrained/{save_name}_epoch_{start_epoch + epochs}.ckpt'
    save_checkpoint(model, start_epoch + epochs, saving_path)

    # Plotting the training and validation losses
    plot_losses(model.TRAINING_LOSSES, model.VALIDATION_LOSSES)

    save_losses_to_csv(model.TRAINING_LOSSES, model.VALIDATION_LOSSES, spectralB, out_dir="/homes/j25lopez/pml/results/",model_name=save_name)


def save_losses_to_csv(training_losses, validation_losses, b_value, out_dir,model_name="turbulence"):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{model_name}_losses_b{b_value}.csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "training_loss", "validation_loss", "b"])
        for i, (train_loss, val_loss) in enumerate(zip(training_losses, validation_losses)):
            writer.writerow([i, train_loss, val_loss, b_value])
