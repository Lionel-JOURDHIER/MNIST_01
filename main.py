import torch
import torch.nn as nn
import torch.nn.functional as F

""" Explication Class VAE
    Variational Autoencoder (VAE) with a classification head for 28x28 grayscale images.

    Architecture:
    -------------
    Encoder:
        - 3 Convolutional layers with ReLU activation, MaxPooling, and Dropout.
        - Outputs feature maps of size (128, 3, 3).
        - Global average pooling produces latent features of size 128.
        - Fully connected layers produce:
            * mu: mean of latent distribution (latent_dim)
            * logvar: log-variance of latent distribution (latent_dim)
        - Classification head flattens features and outputs class logits (latent_dim by default).

    Reparameterization:
        - Samples latent vector z using mu and logvar:
            z = mu + eps * std
        - Allows backpropagation through stochastic sampling.

    Decoder:
        - Fully connected layers map latent vector z to 128*3*3 features.
        - Reshapes features to (128, 3, 3) and passes through ConvTranspose2d layers.
        - Outputs reconstructed image of size (1, 28, 28) with Sigmoid activation.

    Methods:
    --------
    encode(x):
        - Input: x (tensor) of shape (batch_size, 1, 28, 28)
        - Returns: class_out, mu, logvar

    reparameterize(mu, logvar):
        - Input: mu and logvar (tensors)
        - Returns: sampled latent vector z

    decode(z):
        - Input: latent vector z of shape (batch_size, latent_dim)
        - Returns: reconstructed image tensor (batch_size, 1, 28, 28)

    forward(x):
        - Input: x (tensor) of shape (batch_size, 1, 28, 28)
        - Returns: recon_x, class_out, mu, logvar

    Usage:
    ------
    vae = VAE(latent_dim=10)
    recon, class_logits, mu, logvar = vae(x)
"""
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=10):
        super(VAE,self).__init__()   
        #Encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),  # 28x28 → 14x14
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),   # 14x14 → 7x7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # NEW 3rd conv
            nn.ReLU(),
            nn.MaxPool2d(2,2),        # 7x7 → 3x3
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()

        # têtes de distribution latente
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.fc1_layers = nn.Sequential(
            nn.Linear(128*3*3, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, latent_dim)
        )

        #Decoder : 
        self.fc1_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),         # inverse du dernier Linear(128→10)
            nn.ReLU(),
            nn.Linear(128, 128*3*3),    # inverse du premier Linear(1152→128)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),  # 3→7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),    # 14→28
            nn.Sigmoid()  # sortie normalisée entre 0 et 1
        )
    def encode(self, x):
        x = self.conv_layers(x)          # (batch_size, 128, 3, 3)
        
        # For classification
        x_flat = self.flatten(x)  # (batch_size, 128*3*3)
        class_out = self.fc1_layers(x_flat)
        
        # For latent variables: global average pooling to reduce to 128 features
        x_latent = x.mean(dim=[2,3])  # (batch_size, 128)
        mu = self.fc_mu(x_latent)
        logvar = self.fc_logvar(x_latent)
        
        return class_out, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # same shape as std
        return mu + eps * std

    def decode(self, z):
        x = self.fc1_decoder(z)            # (batch_size, 1152)
        x = x.reshape(-1, 128, 3, 3)          # reshape for ConvTranspose
        x = self.decoder(x)                 # (batch_size, 1, 28, 28)
        return x
    
    def forward(self, x):
        class_out, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, class_out, mu, logvar