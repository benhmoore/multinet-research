import io

from PIL import Image
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from torch import nn, optim

from single_net import generate_adjacency_matrix
from multi_net import generate_supra_adjacency_matrix


def train_and_visualize(
    model,
    x,
    y,
    cmap="plasma",
    optimizer=None,
    iterations=120000,
    frame_duration=100,
    interval=3000,
    criterion=nn.L1Loss(),
    output_path="docs/adjacency_matrix_evolution.gif",
):
    """Trains a PyTorch model and visualizes the evolution of the adjacency matrix, outputting a GIF.

    Args:
        model (torch.nn.Module): PyTorch model to train.
        x (torch.Tensor): Input data to the model.
        y (torch.Tensor): Output data to the model.
        cmap (str, optional): Colormap to use for the adjacency matrix. Defaults to "plasma".
        optimizer (PyTorch optimizer, optional): PyTorch optimizer to use for training. Defaults to Adam if None.
        iterations (int, optional): Number of iterations to train the model for. Defaults to 120,000.
        frame_duration (int, optional): Duration of each frame in the GIF in milliseconds. Defaults to 100.
        interval (int, optional): Interval between iterations to plot. Defaults to 3,000.
        criterion (PyTorch loss function, optional): PyTorch loss function to use for training. Defaults to L1Loss.
        output_path (str, optional): Path to save the GIF to. Defaults to "docs/adjacency_matrix_evolution.gif".
    """

    # Use dark background style
    plt.style.use("dark_background")

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.00005)

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
            multi_network_matrices.append(generate_supra_adjacency_matrix(model))
            single_network_matrices.append(generate_adjacency_matrix(model))
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

    # Create frames and keep them in memory
    print("Plotting frames...")
    frames = []
    for iteration, (multi_matrix, single_matrix, loss) in enumerate(
        zip(multi_network_matrices, single_network_matrices, losses)
    ):
        print("Plotting iteration", iteration * interval)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        fig.suptitle(f"Iteration: {iteration*interval}, Loss: {loss:.5f}", fontsize=14)

        im1 = axs[0, 0].imshow(multi_matrix, cmap=cmap, interpolation="none")
        axs[0, 0].set_title(
            f"Multi-Network (Size: {multi_matrix.shape[0]} x {multi_matrix.shape[1]})"
        )
        axs[0, 0].set_xlabel("Nodes")
        axs[0, 0].set_ylabel("Nodes")
        fig.colorbar(im1, ax=axs[0, 0], label="Edge Weight")
        im1.set_clim(min_value_multi_network, max_value_multi_network)

        im2 = axs[0, 1].imshow(single_matrix, cmap=cmap, interpolation="none")
        axs[0, 1].set_title(
            f"Single Network (Size: {single_matrix.shape[0]} x {single_matrix.shape[1]})"
        )
        axs[0, 1].set_xlabel("Nodes")
        axs[0, 1].set_ylabel("Nodes")
        fig.colorbar(im2, ax=axs[0, 1], label="Edge Weight")
        im2.set_clim(min_value_single_network, max_value_single_network)

        # Compute eigenvalues and get their magnitudes
        eigenvalues_multi = np.abs(np.linalg.eigvals(multi_matrix))
        eigenvalues_single = np.abs(np.linalg.eigvals(single_matrix))

        # Create KDE plot for Multi-Network
        plot = sns.kdeplot(eigenvalues_multi, ax=axs[1, 0], fill=True)
        plot.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axs[1, 0].set_title("Multi-Network Eigenvalues")
        axs[1, 0].set_xlabel("Eigenvalues")
        axs[1, 0].set_ylabel("Density")

        # Create KDE plot for Single Network
        plot = sns.kdeplot(eigenvalues_single, ax=axs[1, 1], fill=True)
        plot.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axs[1, 1].set_title("Single Network Eigenvalues")
        axs[1, 1].set_xlabel("Eigenvalues")
        axs[1, 1].set_ylabel("Density")

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
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=[frame_duration] * len(frames[:-1]) + [frame_duration * 20],
        loop=0,
    )
    print(f"GIF generated in {output_path}")
