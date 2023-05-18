import networkx as nx
import pymnet as pm
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt


numnodes = 10
numedge = 3


# Create a pymnet multilayer network object.
net = pm.MultilayerNetwork(aspects=1)

# Create a single layer graph using the networkx.barabasi_albert_graph() function.
layer1 = nx.barabasi_albert_graph(numnodes, numedge)

# Add the single layer graph to the pymnet multilayer network object.
net.add_layer(layer1)

# Repeat steps 3 and 4 for the desired number of layers.
layer2 = nx.barabasi_albert_graph(numnodes, numedge)
net.add_layer(layer2)

# Draw the graph using the spring layout
fig = pm.draw(
    net, layout="spring", show=False
)  # Add 'show=False' to not display the figure immediately

# Save the figure to a PDF file
fig.savefig("net.pdf")
