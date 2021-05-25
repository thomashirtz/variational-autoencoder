import argparse
from variational_autoencoder.dashboard import get_app_dashboard


default_seed = 4
default_num_epochs = 20
default_file_name = None  # 'VAE_20210524_213001.pt'

parser = argparse.ArgumentParser(
    description='PyTorch Variational AutoEncoder',
    usage='Use "python variational-autoencoder --help" for more information'
)
parser.add_argument(
    '--file_name', default=None, type=str, metavar='',
    help='Name used for loading the model\'s weights (default: None)'
)
parser.add_argument(
    '--num-epochs', default=default_num_epochs, type=int, metavar='',
    help='Number of training epochs (default: %(default)s)'
)
parser.add_argument(
    '--seed', default=default_seed, type=int, metavar='',
    help='Seed used for pytorch (default: %(default)s)'
)


args = parser.parse_args()
app = get_app_dashboard(epochs=args.epochs, file_name=args.file_name, seed=args.seed)


if __name__ == "__main__":
    app.run_server(debug=True)
