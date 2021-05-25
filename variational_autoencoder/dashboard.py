import numpy as np
import torch.utils.data
from typing import Optional
from torchvision import datasets
from torchvision import transforms
import pathlib

import plotly.express as px

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from variational_autoencoder import get_vae


def get_app_dashboard(epochs: int = 20, file_name: Optional[str] = None, seed: Optional[int] = None, download: bool = True):
    mnist_path = pathlib.Path(__file__).parents[1].absolute() / 'data'  # todo change in the future
    train_dataset = datasets.MNIST(root=mnist_path, download=download, train=True, transform=transforms.ToTensor())
    vae = get_vae(dataset=train_dataset, epochs=epochs, load_model=True, seed=seed, file_name=file_name)
    print('b')
    with torch.no_grad():
        inputs = train_dataset.data.float()
        zs_tensor = vae.get_z(inputs)
        reconstructed_image = vae.decode(zs_tensor).numpy()
        zs = zs_tensor.numpy()
        original_image = inputs.numpy()
        digits = train_dataset.targets.numpy()

    plot_background_color = "White"  # "LightSteelBlue"
    fig = px.scatter_3d(x=zs[:, 0], y=zs[:, 1], z=zs[:, 2], color=digits, opacity=0.4)
    fig.update_traces(marker={'size': 4})
    fig.update_yaxes(range=[-5, 5])
    fig.update_xaxes(range=[-5, 5])
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=plot_background_color)

    app = dash.Dash(__name__, external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"
    ])

    size_image = 41
    size_graph = [140, 90]
    app.layout = html.Div([
        html.H2("Variational AutoEncoder Dashboard", style={'text-align': 'center'}),
        dbc.Row([
            dbc.Col(width=8, children=[
                html.H3("Latent Space", style={'text-align': 'center'}),
                dcc.Graph(id='latent_space', figure=fig, style={'width': f'{size_graph[0]}vh', 'height': f'{size_graph[1]}vh'}),
            ]),
            dbc.Col(children=[
                html.H3("Original Image", style={'text-align': 'center'}),
                dbc.Row(dcc.Graph(id='original_image'), style={'width': f'{size_image}vh', 'height': f'{size_image}vh'}),
                html.H3("Reconstructed Image", style={'text-align': 'center'}),
                dbc.Row(dcc.Graph(id='reconstructed_image'), style={'width': f'{size_image}vh', 'height': f'{size_image}vh'})
            ])
        ], no_gutters=True, align="center")
    ])

    def get_image_figure(image):
        figure = px.imshow(image)
        figure.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=plot_background_color)
        figure.update(layout_coloraxis_showscale=False)
        return figure

    @app.callback(Output('original_image', 'figure'), Input('latent_space', 'clickData'))
    def get_original_image_figure(data_clicked):
        if data_clicked:
            index = data_clicked['points'][0]['pointNumber']
            image = original_image[index]
        else:
            image = np.zeros((28, 28))
        figure = get_image_figure(image)
        return figure

    @app.callback(Output('reconstructed_image', 'figure'), Input('latent_space', 'clickData'))
    def get_reconstructed_image_figure(data_clicked):
        if data_clicked:
            index = data_clicked['points'][0]['pointNumber']
            image = reconstructed_image[index]
        else:
            image = np.zeros((28, 28))
        figure = get_image_figure(image)
        return figure

    return app
