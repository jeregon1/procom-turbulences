from turbulence.dataset import load_data
from turbulence.train import train

## Training parameters

# Data parameters
nb_images = 500 # Maximum = 500
val_size = 0.2
test_size = 0.2
batch_size = 4

# Training parameters
num_training_epochs = 50
pretrained = True
num_pretrained_epochs = 0

image_folder_path = './velocity_images'
train_loader, val_loader, test_loader = load_data(image_folder_path, nb_images, val_size, test_size, batch_size)

pretrained_model_path = f'./turbulence/pretrained/turbulence_epoch_{num_pretrained_epochs}.ckpt'
train(pretrained_model_path, train_loader, val_loader, epochs=num_training_epochs, pretrained=pretrained)
