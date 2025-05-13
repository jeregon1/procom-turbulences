import os
import torch
import pytorch_lightning as pl
from turbulence.network import Turbulence
from turbulence.utils import plot_losses

class LossLoggerCallback(pl.Callback):
    def __init__(self, model, spectralB, out_dir, model_name):
        super().__init__()
        self.model = model
        self.spectralB = spectralB
        self.out_dir = out_dir
        self.model_name = model_name
        os.makedirs(self.out_dir, exist_ok=True)
        self.filename = os.path.join(self.out_dir, f"{self.model_name}_losses_b{self.spectralB}.csv")
        self.model_save_dir = './turbulence/pretrained/'
        os.makedirs(self.model_save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        import csv
        epoch = trainer.current_epoch
        if epoch % 10 != 0:
            return
        # Save all losses up to current epoch
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "training_loss", "validation_loss", "b"])
            for i, (train_loss, val_loss) in enumerate(zip(self.model.TRAINING_LOSSES, self.model.VALIDATION_LOSSES)):
                writer.writerow([i, train_loss, val_loss, self.spectralB])
        # Save model state
        model_save_path = os.path.join(self.model_save_dir, f"{self.model_name}_epoch_{epoch}.ckpt")
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, model_save_path)

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

    # Add LossLoggerCallback to save losses after every epoch
    loss_logger = LossLoggerCallback(model, spectralB, out_dir="/homes/j25lopez/pml/results/", model_name=save_name)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=5, callbacks=[loss_logger])
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
