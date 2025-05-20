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
        # expand '~' to user home
        self.out_dir = os.path.expanduser(out_dir)
        self.model_name = model_name
        os.makedirs(self.out_dir, exist_ok=True)
        self.filename = os.path.join(self.out_dir, f"{self.model_name}_losses_b{self.spectralB}.csv")
        self.model_save_dir = './turbulence/pretrained/'
        os.makedirs(self.model_save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        import csv
        epoch = trainer.current_epoch
        if epoch % 20 != 0:
            return
        # Save all losses up to current epoch
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "training_loss", "validation_loss", "b"])
            for i, (train_loss, val_loss) in enumerate(zip(self.model.TRAINING_LOSSES, self.model.VALIDATION_LOSSES)):
                writer.writerow([i, train_loss, val_loss, self.spectralB])
        # Save latest model state (overwrite)
        model_save_path = os.path.join(self.model_save_dir, f"{self.model_name}_latest.ckpt")
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
        print(f"Resuming training from epoch {start_epoch} to epoch {epochs}.")
    else : 
        print(f"No pre-trained model found. Training from scratch up to {epochs} epochs.")

    # Prepare callback and exit handler
    loss_logger = LossLoggerCallback(model, spectralB, out_dir="~/pml/results/", model_name=save_name)
    import atexit
    # on exit (normal or via KeyboardInterrupt), save current state
    def _on_exit():
        try:
            epoch = trainer.current_epoch
        except NameError:
            epoch = start_epoch
        ckpt = os.path.join('./turbulence/pretrained', f"{save_name}_latest.ckpt")
        save_checkpoint(model, start_epoch + epoch, ckpt)
        save_losses_to_csv(model.TRAINING_LOSSES, model.VALIDATION_LOSSES, spectralB,
                           out_dir=f"~/pml/results/", model_name=save_name)
    atexit.register(_on_exit)
    
    trainer = pl.Trainer(
        max_epochs= epochs-start_epoch,
        log_every_n_steps=5,
        callbacks=[loss_logger],
        accelerator="gpu",
        devices=1,
        precision=16,
        default_root_dir=os.path.expanduser("~/pml/results/"),
    )
    model.train()
    # ensure Python-level SIGINT is raised (bypass Lightning internal handlers)
    import signal
    def _sigint_handler(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        trainer.fit(model, train_loader, val_loader)
        print("Training finished.")
    except (Exception, KeyboardInterrupt) as e:
        # save on any error
        print(f"Training interrupted by error ({e}). Saving checkpoint and losses...")
        ckpt_path = os.path.join('./turbulence/pretrained', f"{save_name}_latest.ckpt")
        save_checkpoint(model, start_epoch + trainer.current_epoch, ckpt_path)
        save_losses_to_csv(model.TRAINING_LOSSES, model.VALIDATION_LOSSES, spectralB,
                           out_dir="~/pml/results/", model_name=save_name)
        raise
    
    # Saving the checkpoint after training
    # final save as latest
    save_checkpoint(model, start_epoch + epochs, os.path.join('./turbulence/pretrained', f"{save_name}_latest.ckpt"))

    # Plotting the training and validation losses
    plot_losses(model.TRAINING_LOSSES, model.VALIDATION_LOSSES)
    # save loss plot image
    import matplotlib.pyplot as plt
    plt.gcf().savefig(f"~/pml/results/{save_name}_losses.png")

    save_losses_to_csv(model.TRAINING_LOSSES, model.VALIDATION_LOSSES, spectralB, out_dir="~/pml/results/",model_name=save_name)


def save_losses_to_csv(training_losses, validation_losses, b_value, out_dir,model_name="turbulence"):
    import csv
    # expand '~' to user home
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{model_name}_losses_b{b_value}.csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "training_loss", "validation_loss", "b"])
        for i, (train_loss, val_loss) in enumerate(zip(training_losses, validation_losses)):
            writer.writerow([i, train_loss, val_loss, b_value])
