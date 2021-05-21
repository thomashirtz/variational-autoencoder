import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from variational_autoencoder import VAE
from variational_autoencoder import get_device


if __name__ == '__main__':

    epochs = 10
    batch_size = 64
    latent_dimension = 2
    learning_rate = 0.003

    verbose = 200
    download = False

    train_dataset = datasets.MNIST(root='MNIST', download=download, train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='MNIST', download=download, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()

    vae = VAE(shape_input=(784,), latent_dimension=latent_dimension)
    vae = vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):  # noqa
            inputs = inputs.to(device)

            optimizer.zero_grad()
            loss = vae.get_loss(inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % verbose == verbose-1 or i+1 == len(train_loader):
                print(f'[{epoch + 1}/{epochs}, {i + 1}/{len(train_loader)}] loss: {running_loss:.3f}')

    # Displaying the latent space
    with torch.no_grad():
        zs = vae.get_z(train_dataset.data.float().to(device)).data.cpu().numpy()
    targets = train_dataset.targets.cpu().numpy()
    plt.scatter(zs[:, 0], zs[:, 1], c=targets)
    plt.show()
