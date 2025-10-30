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
 
''' Definition open_dataset
    Opens an MNIST dataset file and loads image data into a NumPy array.

    Parameters:
        path (str): Path to the MNIST images file (e.g., "train-images-idx3-ubyte").
        image_size (int): Width/height of each image in pixels (default: 28).
        num_images (int): Number of images to read from the file (default: 10000).

    Returns:
        numpy.ndarray: A 4D array of shape (num_images, image_size, image_size, 1)
                       containing the image data as float32.
    '''
def open_dataset(path:str, image_size:int = 28, num_images:int = 10000) -> np.ndarray:
    with open (path,'rb') as f :
        # Skip header (16 bytes)
        f.read(16)
        # Read images
        buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size)
    return data

''' Definition data_norm
    Normalize image data from 0-255 to 0-1 and convert to a PyTorch tensor with a channel dimension.

    Steps:
    1. Convert the input NumPy array to float32.
    2. Normalize the data to range [0, 1] by dividing by 255.
    3. Convert the normalized array to a PyTorch tensor.
    4. Add a channel dimension at position 1 (unsqueeze).

    Parameters:
    data (np.ndarray): Input image or array with pixel values 0-255.

    Returns:
    torch.Tensor: Normalized tensor of shape (batch?, 1, H, W) suitable for PyTorch models.
    '''
def data_norm(data:np.ndarray) -> torch.Tensor : 
    data_f = data.astype('float32')
    data_n = data_f / 255
    data_t = torch.tensor(data_n, dtype=torch.float32).unsqueeze(1)
    return data_t

''' Definition open_labels
    Load labels from a binary file and return as a PyTorch tensor.

    Parameters:
    path (str): Path to the binary label file.
    num_labels (int): Number of labels to read. Defaults to 10000.

    Returns:
    torch.Tensor: Tensor of labels (dtype=torch.long) suitable for PyTorch loss functions.
    '''
def open_labels(path:str, num_labels: int = 10000) -> torch.Tensor : 
    with open (path,'rb') as f :
        # Skip header (8 bytes)
        f.read(8)
        # Read images
        buf = f.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8)
    labels_t = torch.tensor(labels, dtype=torch.long)  # CrossEntropyLoss exige long
    return labels_t

""" Definition create_validation
    Split dataset into training and validation tensors.

    Args:
        data (torch.Tensor): All images
        labels (torch.Tensor): All labels
        validation_size (int): Number of samples for validation

    Returns:
        data_train, labels_train, data_val, labels_val
    """
def create_validation(data:torch.Tensor, labels: torch.Tensor, validation_size:int = 0) -> torch.Tensor :
    data_size = len(labels)

    if validation_size >= data_size : 
        raise ValueError (f"The Argument validation_size ({validation_size}) must be smaller than dataset size ({data_size})")

    if validation_size == 0:
        return data, labels, None, None
     
    data_train = data[:-validation_size]
    labels_train = labels[0:-validation_size]
    print(len(labels_train))
    data_valid = data[-validation_size:]
    labels_valid = labels[-validation_size:]
    print(len(labels_valid))

    return data_train, labels_train, data_valid, labels_valid

""" Definition create_batch
    Crée un DataLoader PyTorch à partir des tenseurs data et labels.

    Args:
        data (torch.Tensor): Images de forme (N, 1, H, W)
        labels (torch.Tensor): Labels de forme (N,)
        batch_size (int): Taille des batches
        shuffle (bool): Si True, mélange les données

    Returns:
        DataLoader: itérable donnant des batches (images, labels)
    """
def create_batch(data:torch.Tensor, labels: torch.Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

'''Definition save_model
    Save a PyTorch model to disk in the "models" folder.

    Parameters:
    model (nn.Module): The trained PyTorch model to save.
    name (str): Name of the file (without extension) to save the model as.

    Returns:
    None
    '''
def save_model(model: nn.Module, name:str) -> None : 
    torch.save(model, f'models/{name}.pth')

'''Definition load_model
    Load a PyTorch model from a .pth file.

    Parameters:
    path (str): Path to the saved model file (e.g., "models/model_name.pth").

    Returns:
    nn.Module: The loaded PyTorch model.
    '''
def load_model(path:str) -> nn.Module :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, weights_only=False)
    return model.to(device)
