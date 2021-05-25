import torch.utils.data
from torchvision import datasets, transforms

from variational_autoencoder import get_vae
from variational_autoencoder import plot_latent_space
from variational_autoencoder import show_image


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

    dataset = datasets.MNIST(root='../data', download=download, train=True, transform=transforms.ToTensor())

    vae = get_vae(dataset=dataset, epochs=epochs, load_model=load_model, latent_dimension=latent_dimension,
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

        random_inputs = torch.randn((batch_size, latent_dimension))
        random_images = vae.decode(random_inputs).numpy()

    plot_latent_space(zs, digits)

    index = 0
    show_image(original_image, index, f'Original image n°{index}')
    show_image(reconstructed_image, index, f'Reconstructed image n°{index}')

    index = 0
    show_image(random_images, index, f'Random image generated n°{index}')

