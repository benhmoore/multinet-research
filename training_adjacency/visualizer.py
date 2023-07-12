import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from enum import IntEnum
from torch import nn, optim

from single_net import create_graph
from multi_net import model_to_pymnet_plot


def train_and_visualize(
    model,
    x,
    y,
    cmap="plasma",
    optimizer=None,
    iterations=15000,
    frame_duration=50,
    criterion=nn.L1Loss(),
):
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

        if iteration % 100 == 0:
            multi_network_matrices.append(model_to_pymnet_plot(model))
            single_network_matrices.append(create_graph(model))
            losses.append(loss.item())

            print(f"Iteration {iteration} of {iterations} complete.")
            print(f"Loss: {loss.item():.5f}")

    # Get minimum value of all matrices, excluding 0
    min_value_single_network = np.min(
        [np.min(matrix[matrix != 0]) for matrix in single_network_matrices]
    )
    min_value_multi_network = np.min(
        [np.min(matrix[matrix != 0]) for matrix in multi_network_matrices]
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

    # Create frames
    for iteration, (multi_matrix, single_matrix, loss) in enumerate(
        zip(multi_network_matrices, single_network_matrices, losses)
    ):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

        im1 = axs[0].imshow(multi_matrix, cmap=cmap, interpolation="none")
        axs[0].set_title(f"Multi-Network (Iter. {iteration*100}; Loss {loss:.5f})")
        fig.colorbar(im1, ax=axs[0], label="Edge Weight")
        im1.set_clim(
            min_value_multi_network, max_value_multi_network
        )  # consistent scale for each frame

        im2 = axs[1].imshow(single_matrix, cmap=cmap, interpolation="none")
        axs[1].set_title(f"Single Network (Iter. {iteration*100}; Loss {loss:.5f})")
        fig.colorbar(im2, ax=axs[1], label="Edge Weight")
        im2.set_clim(min_value_single_network, max_value_single_network)

        plt.tight_layout()
        plt.savefig(f"frames/frame_{iteration}.png")
        plt.close()

    # Create GIF
    frames = [
        imageio.imread(f"frames/frame_{i}.png")
        for i in range(len(multi_network_matrices))
    ]
    imageio.mimsave("frames/training.gif", frames, loop=15, duration=frame_duration)

    # Delete the individual frames
    for filename in os.listdir("frames"):
        if filename.endswith(".png"):
            os.remove(f"frames/{filename}")
