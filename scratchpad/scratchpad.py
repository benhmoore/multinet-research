import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Define node data with color
def node(id, x, y, z, color):
    return {"id": id, "x": x, "y": y, "z": z, "color": color}

# L1
network_1 = [
    node(1, 1, 1, 1, "blue"),
    node(2, 1, 2, 1, "blue"),
]

# L2
network_2 = [
    node(3, 2, 2, 2, "red"),
    node(4, 2, 1, 2, "red"),
]

# Combine layers
network_sequence = [network_1, network_2]

# Define edges
edges = [
    {"source": 1, "target": 2, "layer": 0},
    {"source": 3, "target": 4, "layer": 1},
    {"source": 1, "target": 3, "layer": "inter"},  # inter-layer connections
]

# Basic layout for app
app.layout = html.Div(
    [
        dcc.Graph(id="3d-scatter-plot"),
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


def find_node(nodes, id):
    return next(node for node in nodes if node["id"] == id)


@app.callback(Output("3d-scatter-plot", "figure"), [Input("time-slider", "value")])
def update_graph(time_step):
    data = []
    # Add edges
    for edge in edges:
        if edge["layer"] == "inter":
            source_node = find_node(network_1, edge["source"])
            target_node = find_node(network_2, edge["target"])
        elif edge["layer"] == time_step:
            source_node = find_node(network_sequence[edge["layer"]], edge["source"])
            target_node = find_node(network_sequence[edge["layer"]], edge["target"])
        else:
            continue

        data.append(
            go.Scatter3d(
                x=[source_node["x"], target_node["x"]],
                y=[source_node["y"], target_node["y"]],
                z=[source_node["z"], target_node["z"]],
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False,
            )
        )

    # Add nodes from all networks at each time step
    for network in network_sequence:
        data.append(
            go.Scatter3d(
                x=[node["x"] for node in network],
                y=[node["y"] for node in network],
                z=[node["z"] for node in network],
                mode="markers",
                marker=dict(
                    size=3,
                    color=[node["color"] for node in network],  # Color based on node data
                    opacity=0.8,
                ),
                showlegend=False,
            )
        )
    
    # Add planes
    x = np.linspace(0, 3, 2)
    y = np.linspace(0, 3, 2)
    xGrid, yGrid = np.meshgrid(y, x)
    z1 = np.ones((2, 2))
    z2 = 2 * np.ones((2, 2))

    # Define surfaces
    data.extend(
        [
            go.Surface(
                x=xGrid,
                y=yGrid,
                z=z1,
                colorscale=[[0, "rgb(0, 0, 255)"], [1, "rgb(0, 0, 255)"]],
                opacity=0.3,
                showscale=False,
            ),
            go.Surface(
                x=xGrid,
                y=yGrid,
                z=z2,
                colorscale=[[0, "rgb(255, 0, 0)"], [1, "rgb(255, 0, 0)"]],
                opacity=0.3,
                showscale=False,
            ),
        ]
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[0, 3], autorange=False),
            yaxis=dict(range=[0, 3], autorange=False),
            zaxis=dict(range=[0, 3], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            uirevision=True,  # Retain user perspective when updating nodes
        )
    )

    return {"data": data, "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)
