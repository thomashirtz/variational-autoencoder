import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from variational_autoencoder import train


if __name__ == '__main__':

    epochs = 1
    batch_size = 128
    latent_dimension = 2
    learning_rate = 0.001

    seed = 9
    verbose = 256
    download = False  # can be set to false after downloading it
    load_model = False
    save_model = True
    checkpoint_directory = '../checkpoints/'
    file_name = None  # 'vae.pt'

    dataset = datasets.MNIST(root='../data/MNIST', download=download, train=True, transform=transforms.ToTensor())

    vae = train(dataset=dataset, epochs=epochs, load_model=load_model, latent_dimension=latent_dimension,
                learning_rate=learning_rate, batch_size=batch_size, seed=seed, verbose=verbose,
                checkpoint_directory=checkpoint_directory, file_name=file_name, save_model=save_model)

    # Plotting some data
    with torch.no_grad():
        inputs = dataset.data.float()
        zs_tensor = vae.get_z(inputs)

        zs = zs_tensor.numpy()
        original_image = inputs.numpy()
        digits = dataset.targets.numpy()
        reconstructed_image = vae.decode(zs_tensor).numpy()

        random_inputs = torch.randn((64, latent_dimension))
        random_images = vae.decode(random_inputs).numpy()

    fig, ax = plt.subplots()
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=digits)
    ax.set_xlim([-5, 5.5])
    ax.set_ylim([-5, 5])
    ax.legend(*scatter.legend_elements(), loc="lower right", title="Digits")
    plt.title('MNIST latent space')
    plt.show()

    index = 1
    plt.imshow(original_image[index])
    plt.show()
    plt.imshow(reconstructed_image[index])
    plt.show()

    index = 1
    plt.imshow(random_images[index])
    plt.show()
