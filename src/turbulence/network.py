import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
import lpips
from turbulence.utils import combined_loss

class Turbulence(pl.LightningModule):

    def __init__(self):
        """
        The __init__ method initializes the layers and structure of the autoencoder.

        Arguments:
        - input_shape: The number of features in the input data. For example, in an image dataset,
                       this would be the total number of pixels (e.g., 28*28 for MNIST).
        """
        super(Turbulence, self).__init__()

        self.best_val_loss = float('inf')
        self.last_best_epoch = 0
        self.TRAINING_LOSSES = []
        self.VALIDATION_LOSSES = []
        self.alpha = 1e-32

        # Encoder part
        self.encoder_linear_layer_1 = nn.Linear(in_features=2048, out_features=1024)
        self.encoder_linear_layer_2 = nn.Linear(in_features=1024, out_features=128)
        self.encoder_linear_layer_3 = nn.Linear(in_features=256, out_features=16)
        self.encoder_linear_layer_12 = nn.Linear(in_features=1024, out_features=128)
        self.encoder_linear_layer_23 = nn.Linear(in_features=256, out_features=16)
        self.encoder_linear_layer_13 = nn.Linear(in_features=1024, out_features=16)

        # Latent space treatment
        self.latent_re = nn.Linear(in_features=48, out_features=48)
        self.latent_im = nn.Linear(in_features=48, out_features=48)

        # Decoder part
        self.decoder_linear_layer_3 = nn.Linear(in_features=48, out_features=128)
        self.decoder_linear_layer_2 = nn.Linear(in_features=256, out_features=342)
        self.decoder_linear_layer_32 = nn.Linear(in_features=48, out_features=128)
        self.decoder_linear_layer_21 = nn.Linear(in_features=256, out_features=341)
        self.decoder_linear_layer_31 = nn.Linear(in_features=48, out_features=341)
        self.decoder_linear_layer_1 = nn.Linear(in_features=1024, out_features=2048)

        # Real part and Imaginary part recovery
        self.re = nn.Linear(in_features=2048, out_features=1024)
        self.im = nn.Linear(in_features=2048, out_features=1024)

        # Initialize weights safely
        self.apply(self.init_weights)

        # Load LPIPS perceptual loss and store as non-module attribute (no .to() issue!)
        lpips_model = lpips.LPIPS(net='alex').eval()
        # Move the model to the same device as the current module
        lpips_model = lpips_model.to(self.device)
        object.__setattr__(self, "_lpips_loss", lpips_model)


    def init_weights(self, m):
        #Applies Kaiming normal initialization to all Linear layers.
        if isinstance(m, nn.Linear):  # Only apply to Linear layers
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, features):
        """
        The forward method defines how the input data flows through the network layers.

        Arguments:
        - features: The input data

        Returns:
        - reconstructed: The reconstructed output, which aims to match the original input data
        """
        ### Preprocessing

        fft = torch.fft.fft(features, n=1024, dim=-1)  # Apply FFT along the last dimension
        fft = torch.cat((fft.real, fft.imag), dim=-1)
        # Shape : (1005,2048)

        ### Encoding

        encode_1 = F.leaky_relu(self.encoder_linear_layer_1(fft))
        encode_2 = F.leaky_relu(self.encoder_linear_layer_2(encode_1))
        encode_12 = F.leaky_relu(self.encoder_linear_layer_12(encode_1))
        encode_13 = F.leaky_relu(self.encoder_linear_layer_13(encode_1))
        encode_out_2 = torch.cat((encode_2, encode_12), dim=-1)
        # Shape : (1005,256)

        encode_3 =  F.leaky_relu(self.encoder_linear_layer_3(encode_out_2))
        encode_23 =  F.leaky_relu(self.encoder_linear_layer_23(encode_out_2))
        encode_out_3 = torch.cat((encode_3, encode_23, encode_13), dim=-1)
        # Shape : (1005,48)

        # Latent Space Treatment
        latent_space_im = self.latent_im(encode_out_3)
        latent_space_re = self.latent_re(encode_out_3)
        self.latent_space_complex = torch.complex(latent_space_re,latent_space_im)

        ### Decoding

        decode_3 =  F.leaky_relu(self.decoder_linear_layer_3(encode_out_3))
        decode_32 =  F.leaky_relu(self.decoder_linear_layer_32(encode_out_3))
        decode_31 =  F.leaky_relu(self.decoder_linear_layer_31(encode_out_3))
        decode_out_3 = torch.cat((decode_3, decode_32), dim=-1)
        # Shape : (1005,256)

        decode_2 =  F.leaky_relu(self.decoder_linear_layer_2(decode_out_3))
        decode_21 =  F.leaky_relu(self.decoder_linear_layer_21(decode_out_3))
        decode_out_2 = torch.cat((decode_2, decode_21, decode_31), dim=-1)
        # Shape : (1005,1024)

        decode_out_1 = self.decoder_linear_layer_1(decode_out_2)
        # Shape : (1005, 2048)

        ### Postprocessing

        re = self.re(decode_out_1)
        im = self.im(decode_out_1)
        out = re + 1j*im
        reconstructed = torch.fft.ifft(out, n=1024, dim=-1)
        # Shape : (1005,1024)

        return reconstructed.real

    """
    def training_step(self, batch, batch_idx):
    """
    """
    Defines a single step in the training loop.

    Arguments:
    - batch: A batch of data provided by the DataLoader
    - batch_idx: Index of the batch

    Returns:
    - loss: The loss calculated for this batch
    """
    """
    x = batch  # Extract the features (input data) from the batch

    # Forward pass through the autoencoder
    x_hat = self(x)

    final_loss = combined_loss(x_hat,x,self,self.alpha)
    final_loss_score = final_loss.item()

    # Taking loss
    self.TRAINING_LOSSES.append(np.log(final_loss_score))

    # Log the training loss (for visualization later)
    self.log('train_loss', final_loss)

    return final_loss
    """
    
    
    def training_step (self, batch, batch_idx) :
        """
        Defines a single step in the training loop.

        Arguments:
        - batch: A batch of data provided by the DataLoader
        - batch_idx: Index of the batch

        Returns:
        - loss: The loss calculated for this batch
        """
        x = batch
        x_hat = self(x)

        final_loss = combined_loss(x_hat, x, self, self.alpha)

        # LPIPS Loss
        x_norm = (x - 0.5) * 2
        x_hat_norm = (x_hat - 0.5) * 2

        # Obtenir le device de LPIPS
        lpips_device = next(self._lpips_loss.parameters()).device
        x_norm = x_norm.to(lpips_device)
        x_hat_norm = x_hat_norm.to(lpips_device)

        lpips_loss_value = self._lpips_loss(x_norm, x_hat_norm).mean()

        lambda_lpips = 0.01

        total_loss = final_loss + lambda_lpips * lpips_loss_value

        total_loss_score = total_loss.item()
        self.TRAINING_LOSSES.append(np.log(total_loss_score))
        self.log('train_loss', total_loss)

        return total_loss



    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Arguments:
        - batch: A batch of data provided by the DataLoader
        - batch_idx: Index of the batch
        """
        x = batch  # Extract the features (input data) from the batch

        # Forward pass through the autoencoder
        x_hat = self(x)

        if self.current_epoch > 0 and self.current_epoch % 250 == 0 and batch_idx == 0:
            if self.alpha > 1e-24:
                self.alpha = 1e-22
            else:
                self.alpha*= 1e3
            print(f"Epoch {self.current_epoch}: Updated alpha to {self.alpha}")

        final_loss = combined_loss(x_hat,x,self,self.alpha)
        final_loss_score = final_loss.item()
        
        # LPIPS Loss
        x_norm = (x - 0.5) * 2
        x_hat_norm = (x_hat - 0.5) * 2

        lpips_device = next(self._lpips_loss.parameters()).device
        x_norm = x_norm.to(lpips_device)
        x_hat_norm = x_hat_norm.to(lpips_device)

        lpips_loss_value = self._lpips_loss(x_norm, x_hat_norm).mean()

        lambda_lpips = 0.01

        # Total validation loss
        final_loss = final_loss + lambda_lpips * lpips_loss_value
        final_loss_score = final_loss.item()

        #  update the a used in combined loss automatically
        if self.current_epoch > 0 and self.current_epoch % 250 == 0 and batch_idx == 0:
            self.last_best_epoch = self.current_epoch
            self.best_val_loss = final_loss_score
            torch.save(self.state_dict(),f"./turbulence/pretrained/best_model.pth")
            print("New Phase of training")

        if np.isnan(np.log(final_loss.item())):
            final_loss = torch.tensor(min(self.VALIDATION_LOSSES))

        # Taking loss
        self.VALIDATION_LOSSES.append(np.log(final_loss_score))

        # Keeping the best model
        if final_loss_score < self.best_val_loss and self.current_epoch - self.last_best_epoch > 1:
            self.last_best_epoch = self.current_epoch
            self.best_val_loss = final_loss_score
            torch.save(self.state_dict(),f"./turbulence/pretrained/best_model.pth")
            print("current epoch = ",self.current_epoch," current_loss = ",final_loss_score)

        # Log the validation loss (for visualization later)
        self.log('val_loss', final_loss)


    def configure_optimizers(self):
        """
        Configures the optimizer to be used for training the model.

        Returns:
        - optimizer: The optimizer (Adam in this case) initialized with the model's parameters
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer