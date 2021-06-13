import argparse
from variational_autoencoder.dashboard import get_app_dashboard


DEFAULT_SEED = 4
DEFAULT_NUM_EPOCHS = 20
DEFAULT_FILE_NAME = None  # 'VAE_20210524_213001.pt'


parser = argparse.ArgumentParser(
    description='PyTorch Variational AutoEncoder',
    usage='Use "python variational-autoencoder --help" for more information'
)
parser.add_argument(
    '--file_name', default=DEFAULT_FILE_NAME, type=str, metavar='',
    help='Name used for loading the model\'s weights (default: None)'
)
parser.add_argument(
    '--num-epochs', default=DEFAULT_NUM_EPOCHS, type=int, metavar='',
    help='Number of training epochs (default: %(default)s)'
)
parser.add_argument(
    '--seed', default=DEFAULT_SEED, type=int, metavar='',
    help='Seed used for pytorch (default: %(default)s)'
)


args = parser.parse_args()
app = get_app_dashboard(
    num_epochs=args.num_epochs,
    file_name=args.file_name,
    seed=args.seed
)


if __name__ == "__main__":
    app.run_server(debug=True)
