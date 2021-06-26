from typing import Optional
import pathlib
import numpy as np

import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import dash
import plotly.express as px
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from variational_autoencoder import get_vae


def get_app_dashboard(
        num_epochs: int = 20,
        file_name: Optional[str] = None,
        seed: Optional[int] = None,
        download: bool = True
):

    mnist_path = pathlib.Path(__file__).parents[1].absolute() / 'data'  # todo change in the future
    train_dataset = datasets.MNIST(root=mnist_path, download=download, train=True, transform=transforms.ToTensor())
    vae = get_vae(dataset=train_dataset, num_epochs=num_epochs, load_model=True, seed=seed, file_name=file_name)

    with torch.no_grad():
        inputs = train_dataset.data.float()
        zs_tensor = vae.get_z(inputs)

        zs = zs_tensor.numpy()
        original_image = inputs.numpy()
        reconstructed_image = vae.decode(zs_tensor).numpy()
        digits = train_dataset.targets.numpy().astype('str')

    app = dash.Dash(__name__, external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"
    ])

    size_image = [350, 295]
    size_graph = [950, 660]
    plot_background_color = "White"  # "LightSteelBlue"

    def get_latent_space_figure():
        figure = px.scatter_3d(x=zs[:, 0], y=zs[:, 1], z=zs[:, 2], color=digits, opacity=0.4)
        figure.update_traces(marker={'size': 4})  # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express color_discrete_sequence=px.colors.sequential.Plasma_r
        figure.update_yaxes(range=[-5, 5])
        figure.update_xaxes(range=[-5, 5])
        figure.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor=plot_background_color)
        figure.update_layout(autosize=False, width=size_graph[0], height=size_graph[1])
        figure.update_layout(legend={'itemsizing': 'constant'})
        return figure
    latent_space_figure = get_latent_space_figure()

    app.layout = html.Div([
        html.H2("Variational AutoEncoder Dashboard", style={'text-align': 'center'}),
        dbc.Row(justify='center', no_gutters=True, children=[
            dbc.Col(width=8, children=[
                html.H3("Latent Space", style={'text-align': 'center'}),
                dcc.Graph(id='latent_space', figure=latent_space_figure),
            ]),
            dbc.Col(width=2, children=[
                html.H3("Original Image", style={'text-align': 'center'}),
                dcc.Graph(id='original_image'),
                html.H3("Reconstructed Image", style={'text-align': 'center'}),
                dcc.Graph(id='reconstructed_image')
            ])
        ])
    ])

    def get_image_figure(image):
        figure = px.imshow(image)
        figure.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=plot_background_color)
        figure.update(layout_coloraxis_showscale=False)
        figure.update_layout(autosize=False, width=size_image[0], height=size_image[1])
        return figure

    def get_dynamic_image_figure(data_clicked, data):
        if data_clicked:
            index = data_clicked['points'][0]['pointNumber']
            image = data[index]
        else:
            image = np.zeros((28, 28))
        return get_image_figure(image)

    @app.callback(Output('original_image', 'figure'), Input('latent_space', 'clickData'))
    def get_original_image_figure(data_clicked):
        return get_dynamic_image_figure(data_clicked, original_image)

    @app.callback(Output('reconstructed_image', 'figure'), Input('latent_space', 'clickData'))
    def get_reconstructed_image_figure(data_clicked):
        return get_dynamic_image_figure(data_clicked, reconstructed_image)

    return app
