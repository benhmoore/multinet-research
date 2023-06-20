# imports
import dash
import random
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import networkx as nx
import ndlib.models.epidemics as ep
from ndlib.models.ModelConfig import Configuration


# Function to build the model
def get_sir_model(graph, num_infected):
    model = ep.SIRModel(graph)
    config = Configuration()
    config.add_model_parameter("beta", 0.8)
    config.add_model_parameter("gamma", 0.01)
    infected_nodes = random.sample(list(graph.nodes()), num_infected)
    config.add_model_initial_configuration("Infected", infected_nodes)
    model.set_initial_status(config)
    return model


# Generate the model iterations
def run_sir_model(model, time_steps):
    return model.iteration_bunch(time_steps)


# network
G1 = nx.erdos_renyi_graph(100, 0.06)
G2 = nx.erdos_renyi_graph(100, 0.06)
time_steps = 4  # Set the time_steps globally

# Assign random positions for the nodes in each network layer
for G in [G1, G2]:
    for node in G.nodes():
        G.nodes[node]["pos"] = (random.uniform(-1, 1), random.uniform(-1, 1))

network_layers = [G1, G2]

app = dash.Dash(__name__)

# Initialize the app layout
app.layout = html.Div(
    [
        html.Label("Initial infected nodes:", style={"font-weight": "bold"}),
        dcc.Input(id="input-infected", type="number", value=1),
        dcc.Store(id="model-store"),
        dcc.Store(id="infected-store", data=1),
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


@app.callback(
    Output("infected-store", "data"),
    Input("input-infected", "value"),
)
def update_infected(num_infected):
    return num_infected


@app.callback(
    Output("3d-scatter-plot", "figure"),
    [Input("time-slider", "value"), Input("infected-store", "data")],
)
def update_graph(time_step, num_infected):
    models = [get_sir_model(layer, num_infected) for layer in network_layers]
    model_results = [run_sir_model(model, time_steps) for model in models]

    data = []

    # Create traces for edges and nodes
    for idx, network in enumerate(network_layers):
        edge_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            line={"width": 0.5, "color": "#888"},
            hoverinfo="none",
            mode="lines",
        )

        node_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            hoverinfo="text",
            marker={
                "showscale": False,
                "colorscale": "Viridis",
                "reversescale": True,
                "color": [],
                "size": 6,
                "opacity": 0.8,
                "line": {"width": 0.5, "color": "#888"},
            },
        )

        # Add edges to trace
        for edge in network.edges():
            x0, y0 = network.nodes[edge[0]]["pos"]
            x1, y1 = network.nodes[edge[1]]["pos"]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)
            edge_trace["z"] += (idx, idx, None)

        # Add nodes to trace
        for node in network.nodes:
            x, y = network.nodes[node]["pos"]
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)
            node_trace["z"] += (idx,)
            if node in model_results[idx][time_step]["status"]:
                status = model_results[idx][time_step]["status"][node]
            else:
                status = 0  # default value
            color = (
                "red" if status == 1 else "blue"
            )  # Color based on the infection status
            node_trace["marker"]["color"] += (color,)

        data.extend((edge_trace, node_trace))

    # Define layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="", showticklabels=False, range=[-1, 1], autorange=False),
            yaxis=dict(title="", showticklabels=False, range=[-1, 1], autorange=False),
            zaxis=dict(title="", showticklabels=False, range=[-1, 1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        )
    )

    return {"data": data, "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)
