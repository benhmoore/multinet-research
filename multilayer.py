from pymnet import *

mnet = MultilayerNetwork(aspects=1)

mnet.add_node(1)
mnet.add_layer("a")

mnet.add_node(2, "b")

mnet[1, "a"][2, "b"] = 1

print(mnet[1, "a"].deg())


fig = draw(
    mnet, layout="spring", show=False
)  # Add 'show=False' to not display the figure immediately

# Save the figure to a PDF file
fig.savefig("multilayer.pdf")
