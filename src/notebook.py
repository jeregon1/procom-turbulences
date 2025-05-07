from turbulence.dataset import load_data
from turbulence.train import train

## Training parameters

# Data parameters
nb_images = 500 # Maximum = 500
val_size = 0.2
test_size = 0.2
batch_size = 64

# Training parameters
num_training_epochs = 500
pretrained = False
num_pretrained_epochs = 0

image_folder_path = './velocity_images'
train_loader, val_loader, test_loader = load_data(image_folder_path, nb_images, val_size, test_size, batch_size)

pretrained_model_path = f'./turbulence/pretrained/turbulence_epoch_{num_pretrained_epochs}.ckpt'

b_values = [0.001, 0.01, 0.1, 0.5]
for b in b_values:
    print(f"\n=== Training with spectral loss b={b} ===\n")
    # Create a new model instance for each run
    train(
        pretrained_model_path,
        train_loader,
        val_loader,
        epochs=50,
        pretrained=pretrained,
        save_name=f"lpips-spectralLoss_b{b}",
        spectralB=b
    )
    