import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=1, output_padding=0),  # 3→7
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 7→14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
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

"""Definition vae_loss
    Computes the loss for a Variational Autoencoder (VAE).

    The loss is a combination of:
    1. Reconstruction loss (Binary Cross-Entropy) between the input and reconstructed images.
    2. Kullback-Leibler (KL) divergence between the learned latent distribution and the standard normal distribution.

    Parameters:
    -----------
    recon_x : torch.Tensor
        Reconstructed image tensor of shape (batch_size, 1, 28, 28), output from the decoder.
    x : torch.Tensor
        Original input image tensor of shape (batch_size, 1, 28, 28).
    mu : torch.Tensor
        Mean of the latent Gaussian distribution (batch_size, latent_dim).
    logvar : torch.Tensor
        Log-variance of the latent Gaussian distribution (batch_size, latent_dim).

    Returns:
    --------
    torch.Tensor
        Scalar tensor representing the total VAE loss (BCE + KLD) summed over the batch.
"""
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

"""Definition train_model
    Train a VAE model with a classification head using joint loss.

    The function optimizes the model using a combination of:
        1. VAE reconstruction + KL divergence loss (vae_loss)
        2. Classification loss (CrossEntropyLoss)
       The classification loss can be optionally weighted.

    Learning rate scheduling:
        - CosineAnnealingLR: smooth decay across epochs
        - ReduceLROnPlateau: reduce LR when validation loss plateaus

    Parameters:
    -----------
    model : nn.Module
        VAE model with classification head
    loader : DataLoader
        Training dataset loader
    val_loader : DataLoader
        Validation dataset loader
    num_epochs : int
        Number of training epochs
    lr : float
        Initial learning rate for Adam optimizer
    factor : float
        Factor to reduce LR on plateau (ReduceLROnPlateau)
    patience : int
        Number of epochs with no improvement before reducing LR

    Returns:
    --------
    model : nn.Module
        Trained model with best validation weights loaded

    Notes:
    ------
    - Uses GPU if available.
    - Joint loss is computed as: total_loss = vae_loss + 0.5 * classification_loss
    - Validation loss and accuracy are printed each epoch.
"""
def train_model(
    model: nn.Module, 
    loader: DataLoader, 
    val_loader: DataLoader,
    num_epochs: int = 5, 
    lr: float = 0.0001, 
    factor=0.5,
    patience=3
    ) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    scheduler1 = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr) 
    scheduler2 = ReduceLROnPlateau(
        optimizer, 
        mode='min',       # monitors validation loss
        factor=factor,       # reduce LR by half
        patience=patience,
        min_lr=lr       # wait 3 epochs without improvement
    )
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            recon_x, outputs, mu, logvar = model(batch_images)

            loss_vae = vae_loss(recon_x, batch_images, mu, logvar)
            loss_cls = criterion(outputs, batch_labels)
            loss = loss_vae + 0.5*loss_cls

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_images.size(0)
        
        scheduler1.step()
        epoch_loss = running_loss / len(loader.dataset)

        val_loss, val_acc = val_model(model, val_loader, criterion=criterion)
        scheduler2.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print('Finished Training')
    return model
    
''' Definition test_model
    Evaluate a PyTorch model on a dataset and return the accuracy.

    Parameters:
    model (nn.Module): The trained PyTorch model to evaluate.
    loader (DataLoader): DataLoader providing the evaluation dataset.

    Returns:
    float: Accuracy of the model on the dataset, as a percentage.
'''
def test_model(model: nn.Module, loader: DataLoader, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # passe en mode évaluation
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():  # désactive gradient pour accélérer
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)[1]                  # forward pass
            _, predicted = torch.max(outputs, 1)    # classe avec probabilité max
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted)
            all_labels.append(labels)

    accuracy = 100 * correct / total
    # Confusion matrix
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:\n", cm)
    return accuracy, cm

''' Definition val_model
    Evaluate a model on a validation set.

    Args:
        model (nn.Module): trained model
        val_loader (DataLoader): validation dataloader
        criterion (nn.Module, optional): loss function (e.g. CrossEntropyLoss).
                                         If None, only accuracy is returned.

    Returns:
        (val_loss, accuracy)
        - val_loss (float or None): average loss over validation set
        - accuracy (float): accuracy in %
    '''
def val_model(model: nn.Module, val_loader: DataLoader, criterion : nn.Module) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # passe en mode évaluation
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # désactive gradient pour accélérer
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)[1]                  # forward pass
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)    # classe avec probabilité max
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = total_loss / total
    accuracy = 100 * correct / total
    return val_loss, accuracy

"""Definition reconstruction_by_index
    Display the reconstruction of a single image by its index in the dataset.

    Parameters:
    -----------
    model : nn.Module
        Trained VAE model
    dataset : torch.Tensor
        Dataset of images, shape (N, 1, 28, 28)
    index : int
        Index of the image to display
    save_path : str or None
        If provided, saves the reconstructed image
"""
def reconstruction_by_index(model: torch.nn.Module, dataset: torch.Tensor, index: int = 0, save_directory: str = None, save_name: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    os.makedirs(f'{save_directory}', exist_ok=True)

    if index < 0 or index >= len(dataset):
        raise ValueError(f"Index {index} out of bounds for dataset of size {len(dataset)}")
    
    image = dataset[index].unsqueeze(0).to(device)  # add batch dimension
    
    with torch.no_grad():
        recon, _, _, _ = model(image)
    
    recon = recon.cpu().squeeze()  # remove batch and channel dimensions
    image = image.cpu().squeeze()
    
    threshold = 0.2  # adjust this between 0.3–0.7 depending on your recon values
    recon_adjusted = torch.sigmoid((recon - 0.5) * 6)
    recon_adjusted[recon_adjusted < threshold] = 0.0

    combined = torch.cat((image, recon, recon_adjusted), dim=1)

    plt.imsave(f'{save_directory}/{save_name}', combined.numpy(), cmap='gray')
    print(f"Image saved to {save_directory}/{save_name}")

if __name__ == "__main__":
    data_test = data_norm(open_dataset('dataset/t10k-images-idx3-ubyte',num_images = 32))
    labels_test = open_labels('dataset/t10k-labels-idx1-ubyte',num_labels = 32)
    model = load_model('models/vae_classification.pth')
   
    for i in range(len(data_test)):
        reconstruction_by_index(
            model=model,
            dataset = data_test,
            index=i,
            save_directory=f'output/images',
            save_name=f'index_{i}_{labels_test[i]}.png'
            )    

""" To train a model
    data = data_norm(open_dataset('dataset/train-images-idx3-ubyte',num_images = 60000))
    labels = open_labels('dataset/train-labels-idx1-ubyte',num_labels = 60000)
    data_train, labels_train, data_valid, labels_valid = create_validation(data,labels,validation_size=10000)

    loader_train = create_batch(data_train, labels_train, batch_size=64, shuffle=True)
    loader_valid = create_batch(data_valid, labels_valid, batch_size=64, shuffle=False)

    data_test = data_norm(open_dataset('dataset/t10k-images-idx3-ubyte',num_images = 10000))
    labels_test = open_labels('dataset/t10k-labels-idx1-ubyte',num_labels = 10000)
    loader_test = create_batch(data_test, labels_test, batch_size=64, shuffle=True)

    model = VAE(latent_dim=10)
    model = train_model(
        model=model, 
        loader = loader_train, 
        val_loader = loader_valid,
        num_epochs = 40, 
        lr = 0.0001,  
        factor=0.5,
        patience=2
    )
    test_model(model,loader=loader_test) 
    save_model(model=model, name='vae_classification')
"""
