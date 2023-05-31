import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
import pymnet as pn

numnodes = 10
numedge = 3

G = nx.barabasi_albert_graph(numnodes, numedge)

# SIR model
model = ep.SIRModel(G)
config = mc.Configuration()  # creates configuration object
config.add_model_parameter(
    "beta", 0.1
)  # infection rate, probability of a susceptible node becoming infected
config.add_model_parameter("gamma", 0.5)  # recovery rate
config.add_model_parameter(
    "fraction_infected", 0.05
)  # fraction of nodes in the network that are initially infected
model.set_initial_status(config)

# Simulation for 10 iterations
iterations = model.iteration_bunch(10)

# Visualization
pos = nx.spring_layout(G)
node_status = [
    model.status[node] for node in range(numnodes)
]  # list that contains status
colors = ["g" if s == 0 else "r" if s == 1 else "b" for s in node_status]

# Create a multilayer network with one layer and one aspect
net = pn.MultilayerNetwork(aspects=1)
layer = "SIR"
aspect = "barabasi_albert"

# Add nodes and edges to the multilayer network from the networkx graph
for node in G.nodes():
    net[node, layer, aspect] = 1  # add node with attribute 1

for edge in G.edges():
    net[edge[0], edge[1], layer, aspect] = 1  # add edge with attribute 1

# Draw the multilayer network using pymnet's visualization tools
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
pn.visuals.draw(
    net,
    ax=ax,
    layout=pos,
    layergap=0.2,
    nodeCoords=True,
    layerColorDict={layer: "black"},
    aspectColorDict={aspect: "black"},
    defaultNodeColor=colors,
)
plt.show()
