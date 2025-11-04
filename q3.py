import numpy as np

import torch
import torch.utils
import torchvision
import matplotlib.pyplot as plt

def loadMNIST():
    # Load the MNIST dataset
    # Return the training and testing datasets
    dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=torchvision.transforms.ToTensor())
    return dataset

def getEncoder():
    # Return the encoder network
    return torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    )

def getDecoder():
    # Return the decoder network
    return torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 784),
        torch.nn.Sigmoid()
    )



def train():
    # Load the MNIST dataset
    dataset = loadMNIST()

    # Get the encoder and decoder networks
    encoder = getEncoder()
    decoder = getDecoder()

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0002)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    batch_size = 10
    # Train the autoencoder
    for epoch in range(5):
        for images, _ in dataLoader:
            images = images.view(batch_size, -1)
            inputs = images + 0.03 * torch.randn(images.size())
            # Forward pass
            encoded = encoder(inputs)
            decoded = decoder(encoded)

            # Compute the loss
            loss = loss_fn(decoded, images)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:', epoch, 'Loss:', loss.item())
    torch.save(encoder, 'encoder.pth')
    torch.save(decoder, 'decoder.pth')

    return encoder, decoder


def plotEncoderMap(encoder, dataset):
    # Plot the 2D map of the encoder
    encoder.eval()
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=True)
    for images, labels in dataLoader:
        images = images.view(200, -1)
        encoded = encoder(images)
        break
    latentSpace = encoded.detach().numpy()
    labels = labels.detach().numpy()
    f = plt.figure()
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    for i in range(200):
        plt.gca().text(latentSpace[i, 0], latentSpace[i, 1], str(labels[i].item()), color=colors[labels[i].item()])
    plt.gca().set_xlim(np.min(latentSpace[:, 0])*1.3, np.max(latentSpace[:, 0])*1.3)
    plt.gca().set_ylim(np.min(latentSpace[:, 1])*1.3, np.max(latentSpace[:, 1])*1.3)
    plt.title('2D map of the encoder Latent Space')
    plt.show()
    return latentSpace

def plotDecoderMap(decoder, latentSpace):
    # Plot the 2D map of the decoder
    decoder.eval()
    x = np.linspace(np.min(latentSpace[:, 0]), np.max(latentSpace[:, 0]), 10).astype(np.float32)
    y = np.linspace(np.min(latentSpace[:, 1]), np.max(latentSpace[:, 1]), 10).astype(np.float32)
    f, axarr = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            z = torch.tensor([[x[i], y[j]]], dtype=torch.float32)
            decoded = decoder(z)
            decoded = decoded.view(28, 28).detach().numpy()
            axarr[9-j, i].imshow(decoded, cmap='gray')
            axarr[9-j, i].axis('off')
    f.tight_layout()
    f.show()
    