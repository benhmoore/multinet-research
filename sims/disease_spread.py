import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import networkx as nx
import ndlib.models.epidemics as ep
from ndlib.models.ModelConfig import Configuration

import numpy as np

from scipy.linalg import block_diag


app = dash.Dash(__name__)


def calculate_supra_laplacian(
    network_layers, interlayer_edges: list, interlayer_weight: float = 1.0
):
    """Calculate the supra-Laplacian matrix of a multiplex network.

    Args:
        network_layers (list[networkx.Graph]): List of network layers.
        interlayer_edges (list[tuple]): List of interlayer edges.
        interlayer_weight (float, optional): Weight of interlayer edges. Defaults to 1.0.

    Returns:
        numpy.ndarray: Supra-Laplacian matrix of the multiplex network.
    """
    n = len(network_layers[0].nodes)  # number of nodes in each layer
    m = len(network_layers)  # number of layers

    # Create aggregated adjacency matrix
    adj_matrices = [nx.adjacency_matrix(G).toarray() for G in network_layers]
    aggregated_adj_matrix = block_diag(*adj_matrices)

    # Create supra-Laplacian matrix from aggregated adjacency matrix
    # The supra-Laplacian matrix is a block diagonal matrix where each
    # block corresponds to a layer in the multiplex network.
    # The size of each block is n x n, where n is the number of
    # nodes in each layer.
    supra_L = (
        nx.laplacian_matrix(nx.from_numpy_array(aggregated_adj_matrix))
        .toarray()
        .astype(float)
    )

    # Update supra-Laplacian to account for the interlayer edges
    for edge in interlayer_edges:
        i, j, layer_i, layer_j = edge

        # Add the interlayer weight to each interlayer edge in the
        # supra-Laplacian matrix using list slicing.
        supra_L[
            layer_i * n : (layer_i + 1) * n, layer_j * n : (layer_j + 1) * n
        ] += interlayer_weight
        supra_L[
            layer_j * n : (layer_j + 1) * n, layer_i * n : (layer_i + 1) * n
        ] += interlayer_weight

    return supra_L


# The multiplex network
G1 = nx.Graph([(1, 2), (2, 3)])
G2 = nx.Graph([(3, 4), (4, 1)])
network_layers = [G1, G2]

# Interlayer edges
interlayer_edges = [(1, 1, 0, 1), (2, 2, 0, 1)]
supra_L = calculate_supra_laplacian(network_layers, interlayer_edges)

# Create SIR model for each layer
model_list = []
for i, G in enumerate(network_layers):
    model = ep.SIRModel(G)

    # Model Configuration
    config = Configuration()
    config.add_model_parameter("beta", 0.01)  # Infectiousness
    config.add_model_parameter("gamma", 0.01)  # Recovery rate

    # Setting the initial status of the nodes
    if i == 0:  # Initial infected node in first layer
        config.add_model_initial_configuration("Infected", [1])  # Node 1 is infected
    else:
        config.add_model_initial_configuration(
            "Susceptible", G.nodes()
        )  # All nodes are susceptible

    model.set_initial_status(config)
    iterations = model.iteration_bunch(4)  # 4 iterations for 4 slider positions
    model_list.append(iterations)


# Preprocess data
network_sequence = []
for t in range(4):
    networks_t = []
    for model in model_list:
        nodes_status = model[t]["status"]
        nodes_data = [
            dict(
                id=i,
                x=np.real(supra_L[i, i]),
                y=np.imag(supra_L[i, i]),
                z=i,
                color="red" if nodes_status[i] == 1 else "blue",
            )
            for i in nodes_status
        ]
        networks_t.append(nodes_data)
    network_sequence.append(networks_t)

app.layout = html.Div(
    [
        dcc.Graph(id="3d-scatter-plot", style={"height": "800px", "width": "800px"}),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(network_sequence) - 1,
            value=0,
            marks={str(i): f"Time {i}" for i in range(len(network_sequence))},
            step=None,
        ),
    ]
)


@app.callback(Output("3d-scatter-plot", "figure"), [Input("time-slider", "value")])
def update_graph(time_step):
    data = []

    for network in network_sequence[time_step]:
        data.append(
            go.Scatter3d(
                x=[node["x"] for node in network],
                y=[node["y"] for node in network],
                z=[node["z"] for node in network],
                mode="markers",
                marker=dict(
                    size=6,
                    color=[node["color"] for node in network],
                    opacity=0.8,
                ),
                showlegend=False,
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            zaxis=dict(range=[0, len(network_sequence)], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            uirevision=True,  # Retain user perspective when updating nodes
        )
    )

    return {"data": data, "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)
