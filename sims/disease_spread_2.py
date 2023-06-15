# imports
import dash
import random
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import networkx as nx
import ndlib.models.epidemics as ep
from ndlib.models.ModelConfig import Configuration
from scipy.linalg import block_diag


# Supra-Laplacian calculator
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


# Function to build the model
def get_sir_model(graph):
    model = ep.SIRModel(graph)
    config = Configuration()
    config.add_model_parameter("beta", 0.8)
    config.add_model_parameter("gamma", 0.01)
    config.add_model_initial_configuration("Infected", [0])
    model.set_initial_status(config)
    return model


# Generate the model iterations
def run_sir_model(model, time_steps):
    iterations = model.iteration_bunch(time_steps)
    return iterations


# network
G1 = nx.erdos_renyi_graph(100, 0.06)
G2 = nx.erdos_renyi_graph(100, 0.06)

# Assign random positions for the nodes in each network layer
for G in [G1, G2]:
    for node in G.nodes():
        G.nodes[node]["pos"] = (random.uniform(-1, 1), random.uniform(-1, 1))

network_layers = [G1, G2]


# Generate models for each layer
models = [get_sir_model(layer) for layer in network_layers]
time_steps = 4

# Run each model and store results
model_results = [run_sir_model(model, time_steps) for model in models]

app = dash.Dash(__name__)

# Initialize the app layout
app.layout = html.Div(
    [
        dcc.Graph(id="3d-scatter-plot", style={"height": "800px", "width": "800px"}),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=time_steps - 1,
            value=0,
            marks={str(i): f"Time {i}" for i in range(time_steps)},
            step=None,
        ),
    ]
)


# App callback function to update the graph
@app.callback(Output("3d-scatter-plot", "figure"), [Input("time-slider", "value")])
def update_graph(time_step):
    data = []

    # Create traces for edges and nodes
    for idx, network in enumerate(network_layers):
        edge_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=False,
                colorscale="Viridis",
                reversescale=True,
                color=[],
                size=6,
                opacity=0.8,
                line=dict(width=0.5, color="#888"),
            ),
        )

        # Add edges to trace
        for edge in network.edges():
            x0, y0 = network.nodes[edge[0]]["pos"]
            x1, y1 = network.nodes[edge[1]]["pos"]
            edge_trace["x"] += tuple([x0, x1, None])
            edge_trace["y"] += tuple([y0, y1, None])
            edge_trace["z"] += tuple([idx, idx, None])

        # Add nodes to trace
        for node in network.nodes:
            x, y = network.nodes[node]["pos"]
            node_trace["x"] += tuple([x])
            node_trace["y"] += tuple([y])
            node_trace["z"] += tuple([idx])  # Z-coordinate based on the layer index
            if node in model_results[idx][time_step]["status"]:
                status = model_results[idx][time_step]["status"][node]
            else:
                status = 0  # Or whatever default value you want for nodes not present
            color = (
                "red" if status == 1 else "blue"
            )  # Color based on the infection status
            node_trace["marker"]["color"] += tuple([color])

        # Add edge_trace and node_trace to data
        data.append(edge_trace)
        data.append(node_trace)

    # Define layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            zaxis=dict(range=[-1, 1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        )
    )

    return {"data": data, "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)
