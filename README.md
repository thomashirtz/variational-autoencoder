# variational-autoencoder

![](dashboard.gif)

Implementation of the vanilla Variational AutoEncoder using Pytorch.

## Usage



### Importing the module

Example of minimal script:
```
from torchvision import datasets
from torchvision import transforms
from variational_autoencoder import train


dataset = datasets.MNIST(root='../data/MNIST', download=True, train=True, transform=transforms.ToTensor())
vae = get_vae(dataset=dataset)
```

The `get_vae` function takes as input a torch dataset among other arguments, 
Examples of 

### MNIST Dashboard 
A dashboard was made for the MNIST dataset using Plotly and Dash. It is located in the __main__.py
```
python3 variational-autoencoder
```
The script will then give the adress of the server on which dash is running
(generally http://127.0.0.1:8050/)

*Help for running the dashboard:*

```
usage: Use "python variational-autoencoder --help" for more information

PyTorch Variational AutoEncoder

optional arguments:
  -h, --help     show this help message and exit
  --file_name    Name used for loading the model's weights (default: None)
  --num-epochs   Number of training epochs (default: 20)
  --seed         Seed used for pytorch (default: 4)
```

## Installation
```
git clone https://github.com/thomashirtz/variational-autoencoder
cd variational-autoencoder
pip install -r requirements.txt
```

## Original Paper

Auto-Encoding Variational Bayes, Kingma et al. [[arxiv]](https://arxiv.org/abs/1312.6114) (January 2013) 
