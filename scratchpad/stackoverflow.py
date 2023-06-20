"""
Plot multi-graphs in 3D.
Reference: https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class LayeredNetworkGraph(object):
    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection="3d")

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()

    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])

    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend(
                [((source, z), (target, z)) for source, target in g.edges()]
            )

    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend(
                [((node, z1), (node, z2)) for node in shared_nodes]
            )

    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.
        # TODO: implement FR in 3D with layer constraints.

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(
            composition, scale=2, *args, **kwargs
        )  # Increase scale factor

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update(
                {
                    (node, z): (*pos[node], z * 2) for node in g.nodes()
                }  # Multiply z by a scaling factor
            )

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        segments = [
            (self.node_positions[source], self.node_positions[target])
            for source, target in edges
        ]
        line_collection = Line3DCollection(segments, linewidths=0.8, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

    def get_extent(self, pad=0.5):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = z * 2 * np.ones_like(U)  # Multiply z by a scaling factor
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                ax.text(
                    *self.node_positions[(node, z)], node_labels[node], *args, **kwargs
                )

    def draw(self):
        self.draw_edges(
            self.edges_within_layers, color="k", alpha=0.2, linestyle="-", zorder=-1
        )
        self.draw_edges(
            self.edges_between_layers, color="k", alpha=0.2, linestyle="--", zorder=-1
        )

        self.ax.set_xlim([-1, 1])  # Adjust these values as needed
        self.ax.set_ylim([-1, 1])  # Adjust these values as needed
        self.ax.set_zlim([0, self.total_layers])  # Adjust these values as needed
        self.ax.set_box_aspect(
            [1, 1, 1]
        )  # Set the aspect ratio of the axes to be equal

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.3, zorder=5)
            self.draw_nodes(
                [node for node in self.nodes if node[1] == z], s=130, zorder=1, alpha=1
            )

        if self.node_labels:
            self.draw_node_labels(
                self.node_labels,
                horizontalalignment="center",
                verticalalignment="center",
                zorder=100,
                fontsize=8,
                color="white",
            )


if __name__ == "__main__":
    # define graphs
    n = 100  # Increase the number of nodes per layer
    k = 2
    g = nx.random_regular_graph(k, n)
    h = nx.random_regular_graph(k, n)
    i = nx.random_regular_graph(k, n)

    node_labels = {nn: str(nn) for nn in range(n)}

    # initialize figure and plot
    fig = plt.figure(figsize=(15, 10))  # Increase figure size
    ax = fig.add_subplot(111, projection="3d")
    LayeredNetworkGraph(
        [g, h, i], node_labels=node_labels, ax=ax, layout=nx.spring_layout
    )

    for o in fig.findobj():
        o.set_clip_on(False)
    ax.set_axis_off()
    plt.show()
