import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import xarray as xr
from IPython.display import HTML


def animate_3d_trajectories(
    data_array,
    *,
    root_nodes=None,
    plot_type="combined",
    rotate=True,
    show_history=True,
    interval=100,
    frame_skip=1,
    rotation_speed=1,
    root_node_color="dimgrey",
    node_color="orange",
    root_node_alpha=0.5,
    node_alpha=0.5,
    save_path=None,
    return_html_anim=True,
    show_plot=False,
    **plot_kwargs,
):
    """
    Create animated 3D plots of trajectory data with configurable display options.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data with dimensions ['time', 'node', 'dim'] where dim=3 for x,y,z coordinates
    root_nodes : array-like, optional
        Boolean array indicating which nodes are root nodes
    plot_type : str, default='combined'
        Type of plot to create:
        - 'combined': all particles in one plot
        - 'subplots': all particles in separate subplots in one figure
        - 'separate': each particle in a separate figure
    rotate : bool, default=True
        Whether to rotate the view during animation
    show_history : bool, default=True
        Whether to show the trajectory history
    interval : int, default=100
        Milliseconds between frames
    frame_skip : int, default=1
        Only animate every nth frame to reduce size
    rotation_speed : float, default=1
        Speed of rotation (degrees per frame)
        Higher values rotate faster, can be negative for opposite direction
    root_node_color : str, default='darkgrey'
        Color for root nodes
    node_color : str, default='orange'
        Color for regular nodes
    root_node_alpha : float, default=0.5
        Alpha (transparency) value for root node trajectories
    node_alpha : float, default=0.5
        Alpha (transparency) value for regular node trajectories
    save_path : str, optional
        If provided, save the animation to this path (e.g. 'animation.gif' or 'animation.mp4')
        Supported formats include .gif, .mp4, and other formats supported by ffmpeg
    return_html_anim : bool, default=True
        If True, returns HTML animation object(s) for display in notebooks
        If False, only saves the animation file if save_path is provided
    show_plot : bool, default=False
        If True, displays the animation using plt.show()
        Only relevant when running from a script (not in notebook)
    **plot_kwargs : dict
        Additional keyword arguments passed to the plot functions

    Returns
    -------
    HTML or list of HTML or None
        If return_html_anim=True: Single or list of HTML animations depending on plot_type
        If return_html_anim=False and save_path provided: None

    Raises
    ------
    ValueError
        If data_array dimensions are incorrect or root_nodes length doesn't match
    """
    # Verify data dimensions
    if data_array.dims != ("time", "node", "dim") or data_array.sizes["dim"] != 3:
        raise ValueError(
            "Data array must have dimensions ('time', 'node', 'dim') with dim=3"
        )

    n_nodes = data_array.sizes["node"]

    def setup_axis(ax, node_idx=None):
        """Helper function to setup axis properties with individual limits"""
        if node_idx is not None:
            # Get data range for specific node
            node_data = data_array.isel(node=node_idx)
            x_min, x_max = node_data.min().values, node_data.max().values
            margin = (x_max - x_min) * 0.1
        else:
            # Get global data range for combined plot
            x_min, x_max = data_array.min().values, data_array.max().values
            margin = (x_max - x_min) * 0.1

        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([x_min - margin, x_max + margin])
        ax.set_zlim([x_min - margin, x_max + margin])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.ioff()

    if root_nodes is not None and len(root_nodes) != data_array.sizes["node"]:
        raise ValueError("root_nodes length must match number of nodes")

    # Calculate frame indices with skipping
    frame_indices = range(0, data_array.sizes["time"], frame_skip)

    def save_animation(anim, save_path):
        """Helper function to save animation with proper writer setup"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(save_path)
        if output_dir:  # Only create directory if path contains a directory part
            os.makedirs(output_dir, exist_ok=True)

        if save_path.endswith(".gif"):
            writer = "pillow"
        else:
            try:
                # Try to use ffmpeg writer with specific settings
                writer = animation.FFMpegWriter(
                    fps=1000 / interval,  # Convert interval to fps
                    metadata=dict(artist="Me"),
                    bitrate=2000,
                )
            except Exception as e:
                raise ValueError(
                    "Failed to create FFMpegWriter. Make sure ffmpeg is installed. "
                ) from e

        try:
            anim.save(save_path, writer=writer)
        except Exception as e:
            raise ValueError(
                f"Failed to save animation to {save_path}: {str(e)}"
            ) from e

    if plot_type == "combined":
        # Single plot with all particles
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.clear()
            setup_axis(ax)

            # Plot all particles
            current_points = data_array[frame].values

            if root_nodes is not None:
                # Plot non-root nodes
                non_root_points = current_points[~root_nodes]
                if len(non_root_points) > 0:
                    ax.scatter(
                        non_root_points[:, 0],
                        non_root_points[:, 1],
                        non_root_points[:, 2],
                        c=node_color,
                        marker="o",
                        label="Regular nodes",
                        **plot_kwargs,
                    )

                # Plot root nodes
                root_points = current_points[root_nodes]
                if len(root_points) > 0:
                    ax.scatter(
                        root_points[:, 0],
                        root_points[:, 1],
                        root_points[:, 2],
                        c=root_node_color,
                        marker="*",
                        s=200,
                        label="Root nodes",
                        **plot_kwargs,
                    )
                ax.legend()
            else:
                ax.scatter(
                    current_points[:, 0],
                    current_points[:, 1],
                    current_points[:, 2],
                    c=node_color,
                    marker="o",
                    **plot_kwargs,
                )

            if show_history and frame > 0:
                history = data_array[0:frame].values
                for n in range(n_nodes):
                    color = (
                        root_node_color
                        if root_nodes is not None and root_nodes[n]
                        else node_color
                    )
                    alpha = (
                        root_node_alpha
                        if root_nodes is not None and root_nodes[n]
                        else node_alpha
                    )
                    ax.plot(
                        history[:, n, 0],
                        history[:, n, 1],
                        history[:, n, 2],
                        c=color,
                        alpha=alpha,
                        linestyle=(
                            "--" if root_nodes is not None and root_nodes[n] else "-"
                        ),
                        **plot_kwargs,
                    )

            if rotate:
                ax.view_init(elev=30, azim=frame * rotation_speed % 360)
            ax.set_title(f"Time step: {frame * frame_skip}")

        anim = animation.FuncAnimation(
            fig, update, frames=frame_indices, interval=interval
        )

        if save_path:
            save_animation(anim, str(save_path))

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return HTML(anim.to_jshtml()) if return_html_anim else None

    elif plot_type == "subplots":
        # Multiple subplots in one figure
        n_cols = min(3, n_nodes)
        n_rows = (n_nodes + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        axes = [
            fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
            for i in range(n_nodes)
        ]

        def update(frame):
            for i, ax in enumerate(axes):
                ax.clear()
                setup_axis(ax, node_idx=i)

                current_point = data_array[frame, i].values
                is_root = root_nodes is not None and root_nodes[i]

                ax.scatter(
                    current_point[0],
                    current_point[1],
                    current_point[2],
                    c=root_node_color if is_root else node_color,
                    marker="*" if is_root else "o",
                    s=200 if is_root else 100,
                    label="Root node" if is_root else "Regular node",
                )

                if show_history and frame > 0:
                    history = data_array[0:frame, i].values
                    color = root_node_color if is_root else node_color
                    alpha = root_node_alpha if is_root else node_alpha
                    linestyle = "--" if is_root else "-"
                    ax.plot(
                        history[:, 0],
                        history[:, 1],
                        history[:, 2],
                        c=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        **plot_kwargs,
                    )

                if rotate:
                    ax.view_init(elev=30, azim=frame * rotation_speed % 360)
                node_type = "Root" if is_root else "Node"
                ax.set_title(f"{node_type} {i}\nTime step: {frame * frame_skip}")
                if is_root:
                    ax.legend()

        anim = animation.FuncAnimation(
            fig, update, frames=frame_indices, interval=interval
        )

        if save_path:
            save_animation(anim, str(save_path))

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return HTML(anim.to_jshtml()) if return_html_anim else None

    elif plot_type == "separate":
        # Separate figure for each node
        animations = []
        for node in range(n_nodes):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            def update(frame, node=node):
                ax.clear()
                setup_axis(ax, node_idx=node)

                current_point = data_array[frame, node].values
                is_root = root_nodes is not None and root_nodes[node]

                ax.scatter(
                    current_point[0],
                    current_point[1],
                    current_point[2],
                    c=root_node_color if is_root else node_color,
                    marker="*" if is_root else "o",
                    s=200 if is_root else 100,
                    **plot_kwargs,
                )

                if show_history and frame > 0:
                    history = data_array[0:frame, node].values
                    color = root_node_color if is_root else node_color
                    alpha = root_node_alpha if is_root else node_alpha
                    linestyle = "--" if is_root else "-"
                    ax.plot(
                        history[:, 0],
                        history[:, 1],
                        history[:, 2],
                        c=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        **plot_kwargs,
                    )

                if rotate:
                    ax.view_init(elev=30, azim=frame * rotation_speed % 360)
                node_type = "Root" if is_root else "Node"
                ax.set_title(f"{node_type} {node}\nTime step: {frame * frame_skip}")

            anim = animation.FuncAnimation(
                fig, update, frames=frame_indices, interval=interval
            )

            if save_path:
                base, ext = os.path.splitext(save_path)
                node_save_path = f"{base}_node{node}{ext}"
                save_animation(anim, str(node_save_path))

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            if return_html_anim:
                animations.append(HTML(anim.to_jshtml()))

        return animations if return_html_anim else None

    else:
        raise ValueError("plot_type must be one of: 'combined', 'subplots', 'separate'")


def plot_trajectories(
    data_array,
    root_nodes=None,
    save_path=None,
    root_node_color="dimgrey",
    node_color="orange",
    **plot_kwargs,
) -> xr.plot.FacetGrid:
    """
    Create faceted line plots for trajectory data with root nodes highlighted.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data with dimensions ['time', 'node', 'dim']
    root_nodes : array-like, optional
        Boolean array indicating which nodes are root nodes
    save_path : str, optional
        If provided, save the plot to this path (e.g. 'plot.png' or 'plot.pdf')
    root_node_color : str, optional
        Color to use for root nodes
    **plot_kwargs : dict
        Additional keyword arguments passed to xarray's plot.line method

    Returns
    -------
    xarray.plot.FacetGrid
        The facet grid object containing the plots
    """
    # Create base plot with facets
    plots = data_array.plot.line(
        x="time", col="node", row="dim", color=node_color, **plot_kwargs
    )
    if root_nodes is not None:
        # Update styling for each subplot based on root node status
        for i, ax in enumerate(plots.axs.flat):
            node_num = i % data_array.sizes["node"]
            if root_nodes[node_num]:
                # Update root node appearance
                line = ax.get_lines()[0]
                if root_node_color is not None:
                    line.set_color(root_node_color)

                # Update title to indicate root node, but only on the top row
                if i < data_array.sizes["node"]:  # Only for the top row
                    title = ax.get_title()
                    c = line.get_color()
                    ax.set_title(f"Root {title}", color=c)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plots


def format_vector_values(
    values: np.ndarray | torch.Tensor, precision: int = 2, max_elements: int = 3
) -> str:
    """
    Format vector values for display in nodes.

    Parameters
    ----------
    values : array-like
        Vector values to format
    precision : int, default=2
        Number of decimal places to show
    max_elements : int, default=3
        Maximum number of elements to display before truncating

    Returns
    -------
    str
        Formatted vector values string as "[val1, val2, val3, ...]"
    """
    if values is None:
        return ""

    # Convert to numpy if needed
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    # Format each value with specified precision
    formatted = [f"{v:.{precision}f}" for v in values[:max_elements]]

    # Add ellipsis if there are more elements
    if len(values) > max_elements:
        formatted.append("...")

    return "[" + ", ".join(formatted) + "]"


def plot_scm(
    G,
    node_values=None,
    node_vectors=None,
    pos=None,
    ax=None,
    title="Structural Causal Model",
    figsize=(10, 8),
    show_vectors=True,
    vector_precision=2,
    max_vector_elements=3,
    node_size=700,
    font_size=8,
    root_nodes=None,
) -> tuple[mpl.axes.Axes, dict]:
    """
    Plot a Structural Causal Model with optional node values and vector annotations.

    This function visualizes a directed graph representing a Structural Causal Model (SCM),
    with customizable node colors, sizes, and vector annotations. It supports highlighting
    root nodes and displaying vector values within nodes.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph representing the Structural Causal Model
    node_values : dict or array-like, optional
        Values for each node to be represented by color
    node_vectors : dict or array-like, optional
        Vector values for each node to be displayed as text
    pos : dict, optional
        Node positions for the graph layout. If None, spring layout is used
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes for plotting. If None, a new figure is created
    title : str, default="Structural Causal Model"
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    show_vectors : bool, default=True
        Whether to display vector values in nodes
    vector_precision : int, default=2
        Number of decimal places to show in vector values
    max_vector_elements : int, default=3
        Maximum number of vector elements to display before truncating
    node_size : int, default=700
        Size of nodes in the visualization
    font_size : int, default=8
        Font size for node labels and vector values
    root_nodes : array-like, optional
        Boolean array indicating which nodes are root nodes. If provided, root nodes will be colored dimgrey
        and other nodes will be colored orange.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes with the plot
    dict
        Node positions used in the layout

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 3)])
    >>> root_nodes = np.array([True, False, False, False])
    >>> node_vectors = np.random.randn(4, 3)  # 4 nodes with 3D vectors
    >>> ax, pos = plot_scm(G, node_vectors=node_vectors, root_nodes=root_nodes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if pos is None:
        pos = nx.spring_layout(G)

    # Handle node colors based on root_nodes if provided
    if root_nodes is not None:
        # Convert root_nodes to a list of colors in the same order as G.nodes()
        node_colors = ["grey" if root_nodes[node] else "orange" for node in G.nodes()]
    else:
        node_colors = node_values if node_values is not None else "lightblue"

    # Draw the graph structure
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_colors,
    )

    # Categorize edges
    regular_only_edges = []
    lagged_only_edges = []
    both_edges = []

    # Create a dictionary to track edge types between node pairs
    edge_types = {}
    for u, v, d in G.edges(data=True):
        edge_key = (u, v)
        is_lagged = d.get("label") == "lagged"

        if edge_key not in edge_types:
            edge_types[edge_key] = {"regular": False, "lagged": False}

        if is_lagged:
            edge_types[edge_key]["lagged"] = True
        else:
            edge_types[edge_key]["regular"] = True

    # Categorize edges based on their types
    for edge_key, types in edge_types.items():
        if types["regular"] and types["lagged"]:
            both_edges.append(edge_key)
        elif types["regular"]:
            regular_only_edges.append(edge_key)
        elif types["lagged"]:
            lagged_only_edges.append(edge_key)

    # Draw regular-only edges
    if regular_only_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=regular_only_edges,
            arrowsize=20,
            width=2,
            connectionstyle="arc3,rad=0.1",
        )

    # Draw lagged-only edges with dashed line style
    if lagged_only_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=lagged_only_edges,
            arrowsize=20,
            width=2,
            connectionstyle="arc3,rad=0.1",
            style="dashed",
        )

    # Draw edges that are both regular and lagged with two arrows
    if both_edges:
        # Draw lagged part with dashed line and different curve
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=both_edges,
            arrowsize=20,
            width=2,
            connectionstyle="arc3,rad=0.15",
            style="-.",  # More curved
        )

    # Draw node labels
    if show_vectors and node_vectors is not None:
        # Create custom labels with node number and vector values
        labels = {}
        for node in G.nodes():
            if isinstance(node_vectors, dict) and node in node_vectors:
                vector = node_vectors[node]
            elif isinstance(node_vectors, (np.ndarray, torch.Tensor)) and node < len(
                node_vectors
            ):
                vector = node_vectors[node]
            else:
                vector = None

            if vector is not None:
                formatted = format_vector_values(
                    vector, vector_precision, max_vector_elements
                )
                labels[node] = f"{node}\n{formatted}"
            else:
                labels[node] = f"{node}"

        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)
    else:
        nx.draw_networkx_labels(G, pos, ax=ax)

    ax.set_title(title)
    ax.axis("off")

    return ax, pos

def plot_3d_trajectories(
    data_array,
    root_nodes=None,
    save_path=None,
    root_node_color="dimgrey",
    node_color="orange",
    figsize=(15, 5),
    line_alpha=1.0,
    show_grid=False,
    show_background=True,
    **plot_kwargs,
) -> tuple[mpl.axes.Axes, ...]:
    """
    Create a faceted plot showing the 3D trajectory development over time, with each node
    shown in a separate column.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data with dimensions ['time', 'node', 'dim'] where dim=3 for x,y,z coordinates
    root_nodes : array-like, optional
        Boolean array indicating which nodes are root nodes
    save_path : str, optional
        If provided, save the plot to this path (e.g. 'plot.png' or 'plot.pdf')
    root_node_color : str, default='dimgrey'
        Color to use for root nodes
    node_color : str, default='orange'
        Color to use for regular nodes
    figsize : tuple, default=(15, 5)
        Figure size (width, height) in inches
    line_alpha : float, default=1.0
        Alpha (transparency) value for trajectory lines
    show_grid : bool, default=False
        Whether to show grid lines in the 3D plots
    show_background : bool, default=True
        Whether to show the background panes in the 3D plots
    **plot_kwargs : dict
        Additional keyword arguments passed to the plot functions

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The matplotlib axes objects for each subplot
    """
    n_nodes = data_array.sizes["node"]

    # Create figure with nodes as columns
    fig, axes = plt.subplots(1, n_nodes, figsize=figsize, subplot_kw={'projection': '3d'})
    if n_nodes == 1:
        axes = np.array([axes])  # Ensure axes is always an array

    # Get the final state
    final_state = data_array.isel(time=-1).values

    for node, ax in enumerate(axes):
        is_root = root_nodes is not None and root_nodes[node]
        node_color_use = root_node_color if is_root else node_color
        linestyle = "-" if is_root else "-"
        marker = "*" if is_root else "o"
        marker_size = 200 if is_root else 100

        trajectory = data_array.sel(node=node).values

        # Plot trajectory
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            c=node_color_use,
            alpha=line_alpha,
            linestyle=linestyle,
            **plot_kwargs,
        )

        # Plot final state point
        ax.scatter(
            [final_state[node, 0]],
            [final_state[node, 1]],
            [final_state[node, 2]],
            c=node_color_use,
            marker=marker,
            s=marker_size,
            **plot_kwargs,
        )

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Set title based on node type
        node_type = "Root node" if is_root else "Node"
        ax.set_title(f"{node_type} {node}")

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Control grid visibility
        ax.grid(show_grid)
        # Control background panes visibility
        if not show_background:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            
            # Hide axis lines, labels, and ticks
            ax.set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return tuple(axes)
