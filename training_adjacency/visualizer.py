import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim

from PIL import Image
import io

from single_net import create_graph
from multi_net import model_to_pymnet_plot


def train_and_visualize(
    model,
    x,
    y,
    cmap="plasma",
    optimizer=None,
    iterations=8000,
    frame_duration=80,
    interval=100,
    criterion=nn.L1Loss(),
):
    # Use dark background style
    plt.style.use("dark_background")

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    single_network_matrices = []
    multi_network_matrices = []

    losses = []
    for iteration in range(iterations):
        optimizer.zero_grad()  # zero the gradient buffers

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()  # Does the update

        if iteration % interval == 0:
            multi_network_matrices.append(model_to_pymnet_plot(model))
            single_network_matrices.append(create_graph(model))
            losses.append(loss.item())

            print(f"Iteration {iteration} of {iterations} complete.")
            print(f"Loss: {loss.item():.5f}")

    # Get minimum value of all matrices, excluding 0
    min_value_single_network = np.min(
        [np.min(matrix[matrix > 0]) for matrix in single_network_matrices]
    )
    min_value_multi_network = np.min(
        [np.min(matrix[matrix > 0]) for matrix in multi_network_matrices]
    )

    # Get maximum value of all matrices
    max_value_multi_network = np.max(
        [np.max(matrix) for matrix in multi_network_matrices]
    )
    max_value_single_network = np.max(
        [np.max(matrix) for matrix in single_network_matrices]
    )

    # Create directory for frames if it doesn't exist
    if not os.path.exists("frames"):
        os.makedirs("frames")

    # Create frames and keep them in memory
    frames = []
    for iteration, (multi_matrix, single_matrix, loss) in enumerate(
        zip(multi_network_matrices, single_network_matrices, losses)
    ):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        fig.suptitle(f"Iteration: {iteration*interval}, Loss: {loss:.5f}", fontsize=14)

        im1 = axs[0, 0].imshow(multi_matrix, cmap=cmap, interpolation="none")
        axs[0, 0].set_title("Multi-Network")
        axs[0, 0].set_xlabel("Nodes")
        axs[0, 0].set_ylabel("Nodes")
        fig.colorbar(im1, ax=axs[0, 0], label="Edge Weight")
        im1.set_clim(min_value_multi_network, max_value_multi_network)

        im2 = axs[0, 1].imshow(single_matrix, cmap=cmap, interpolation="none")
        axs[0, 1].set_title("Single Network")
        axs[0, 1].set_xlabel("Nodes")
        axs[0, 1].set_ylabel("Nodes")
        fig.colorbar(im2, ax=axs[0, 1], label="Edge Weight")
        im2.set_clim(min_value_single_network, max_value_single_network)

        # Getting non-zero weights
        non_zero_multi_weights = multi_matrix[multi_matrix.nonzero()].ravel()
        non_zero_single_weights = single_matrix[single_matrix.nonzero()].ravel()

        (n, bins, patches) = axs[1, 0].hist(non_zero_multi_weights, bins=50)
        axs[1, 0].set_title("Multi-Network Histogram")
        axs[1, 0].set_xlabel("Edge Weight")
        axs[1, 0].set_ylabel("Frequency")

        (n, bins, patches) = axs[1, 1].hist(non_zero_single_weights, bins=50)
        axs[1, 1].set_title("Single Network Histogram")
        axs[1, 1].set_xlabel("Edge Weight")
        axs[1, 1].set_ylabel("Frequency")

        plt.tight_layout()

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Use PIL to open the image and convert to RGB
        image = Image.open(buf).convert("RGB")
        frames.append(image)

        plt.close()

    # Save the frames as a GIF
    print("Generating GIF...")

    frames[0].save(
        "frames/training.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration,
        loop=0,
    )

    print("GIF generated in frames/training.gif")

    # Delete the individual frames
    for filename in os.listdir("frames"):
        if filename.endswith(".png"):
            os.remove(f"frames/{filename}")
