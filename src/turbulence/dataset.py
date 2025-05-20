import os
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class VelocityDataset(Dataset):
    def __init__(self, image_files, transform=None):
        """
        Args:
            image_files (list): List of images paths
            transform (callable, optional): Transformation to apply on image
        """
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        """
        Return the number of images
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        charge and return an image at a specific index
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image

def create_data_loaders(data, test_size ,valid_size ,batch_size):
    #split training and test
    len_train_set = int(len(data)*(1-test_size))
    train_set , test_set = random_split(data, [len_train_set,len(data)-len_train_set])

    #split training and validation
    len_val_set = int(len_train_set*valid_size)
    train_subset , val_subset = random_split(train_set, [len_train_set-len_val_set,len_val_set])

    #creating test loader
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    #creating training dataloader
    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    #creating valid_loader
    valid_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, valid_loader, test_loader

def load_data(image_folder_path, nb_images=100, test_size=0.2, val_size=0.2, batch_size=4):
    # List of files paths
    image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.png')]
    random.shuffle(image_files)
    image_files = image_files[:nb_images]
    image_dataset = VelocityDataset(image_files, transform=transform)
    train_loader, val_loader, test_loader = create_data_loaders(image_dataset , test_size ,val_size ,batch_size)
    return train_loader, val_loader, test_loader

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])