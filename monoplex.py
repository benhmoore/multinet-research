from pymnet import *

# Create a monoplex network
net = MultilayerNetwork(aspects=0)

# Add nodes to the network
net.add_node(1)
net.add_node(2)

# Connect nodes with edges where the edge weight is 1
net[1, 2] = 1

# You can also connect nodes that you have not explicitly created
net[1, 3] = 2

print(net[3].deg())  # == 1
