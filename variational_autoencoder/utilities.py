import torch
import torch.utils.data
from typing import Optional
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from variational_autoencoder import VAE


def get_device(overwrite: Optional[bool] = None, device: str = 'cuda:0', fallback_device: str = 'cpu') -> torch.device:
    use_cuda = torch.cuda.is_available() if overwrite is None else overwrite
    return torch.device(device if use_cuda else fallback_device)


def get_file_name(extension: str = '.pt', *arguments, **keyword_arguments) -> str:
    name = f'VAE_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if arguments:
        for argument in arguments:
            name += f'_{str(argument)}'
    if keyword_arguments:
        for argument_name, argument_value in keyword_arguments.items():
            name += f'_{argument_name}{str(argument_value)}'
    return name + extension  # todo edit extension handling


def get_vae(dataset: torch.utils.data.Dataset, num_epochs: int = 1, batch_size: int = 128, latent_dimension: int = 3,
            learning_rate: float = 0.001, verbose: int = 256, seed: Optional[int] = None, save_model: bool = True,
            load_model: bool = False, file_name: Optional[str] = None, checkpoint_directory: str = '../checkpoints/'):

    if seed:
        torch.manual_seed(seed)

    device = get_device()

    checkpoint_path = Path(checkpoint_directory) / (file_name if file_name is not None else get_file_name())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    shape = tuple(dataset[0][0].shape[1:])

    vae = VAE(shape_input=shape, latent_dimension=latent_dimension)
    vae = vae.to(device)

    if load_model and file_name:
        vae.load_state_dict(torch.load(checkpoint_path))

    else:
        print(f'Start training {checkpoint_path}')
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):  # noqa
                inputs = inputs.to(device)

                optimizer.zero_grad()
                loss = vae.get_loss(inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % verbose == verbose-1 or i+1 == len(train_loader):
                    print(f'Epoch: [{epoch + 1}/{num_epochs}] Batch: [{i + 1}/{len(train_loader)}] Loss: {running_loss / i:.3f}')

        if save_model:
            torch.save(vae.state_dict(), checkpoint_path)
            print(f'Variational AutoEncoder Saved to {checkpoint_path}')

    return vae.cpu()


def plot_latent_space(zs, digits):
    fig, ax = plt.subplots()
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=digits)
    ax.set_xlim([-5, 5.5])
    ax.set_ylim([-5, 5])
    ax.legend(*scatter.legend_elements(), loc="lower right", title="Digits")
    plt.title('MNIST latent space')
    plt.show()


def show_image(data, index, title):
    plt.imshow(data[index])
    plt.title(title)
    plt.show()

