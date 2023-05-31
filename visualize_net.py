from pymnet import *

# Create a random graph with 10 nodes and a connection probability of 0.4 for each layer
net = er(3, [0.4, 0.4])

# Draw the graph using the spring layout
fig = draw(
    net, layout="spring", show=False
)  # Add 'show=False' to not display the figure immediately

# Save the figure to a PDF file
fig.savefig("net.pdf")
