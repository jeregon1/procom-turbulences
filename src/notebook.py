from turbulence.dataset import load_data
from turbulence.train import train

## Training parameters

# Data parameters
nb_images = 500 # Maximum = 500
val_size = 0.2
test_size = 0.2
batch_size = 64

# Training parameters
num_training_epochs = 750
pretrained = False
num_pretrained_epochs = 0
b=0.2
name = f"spectralLoss_b{b}"

image_folder_path = './velocity_images'
train_loader, val_loader, test_loader = load_data(image_folder_path, nb_images, val_size, test_size, batch_size)

# pretrained_model_path = f'./turbulence/pretrained/{name}_epoch_{num_pretrained_epochs}.ckpt'
pretrained_model_path = f'./lightning_logs/version_54/checkpoints/epoch=396.ckpt'

train(
    pretrained_model_path,
    train_loader,
    val_loader,
    epochs=num_training_epochs,
    pretrained=pretrained,
    save_name=name,
    spectralB=b
)

# b_values = [1e-5, 1e-4]
# for b in b_values:
#     print(f"\n=== Training with spectral loss b={b} ===\n")
#     # Create a new model instance fo
#     train(
#         pretrained_model_path,
#         train_loader,
#         val_loader,
#         epochs=num_training_epochs,
#         pretrained=pretrained,
#         save_name=f"spectralLoss_b{b}",
#         spectralB=b
#     )
    