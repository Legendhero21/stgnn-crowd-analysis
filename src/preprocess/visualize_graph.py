import os
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import cv2

# Allow PyTorch to unpickle PyG Data objects
torch.serialization.add_safe_globals([Data])

def visualize_frame(graph, save_path=None):
    """Visualize one PyG graph (frame) with nodes, edges, density colors, velocity arrows."""
    # Load image
    img = cv2.imread(graph.image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {graph.image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot base image
    plt.figure(figsize=(12, 9))
    plt.imshow(img)

    # Node coords, velocities, densities
    coords = graph.x[:, :2].numpy()
    velocities = graph.x[:, 2:4].numpy()
    densities = graph.x[:, 4].numpy()

    # Draw edges
    for (src, dst) in graph.edge_index.T:
        x1, y1 = coords[src]
        x2, y2 = coords[dst]
        plt.plot([x1, x2], [y1, y2], 'y-', alpha=0.2, linewidth=0.5)

    # Draw nodes (colored by density)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=densities,
                     cmap='viridis', s=30, edgecolors='k')

    # Draw velocity arrows
    for (x, y), (dx, dy) in zip(coords, velocities):
        if dx != 0 or dy != 0:  # skip zero velocity
            plt.arrow(x, y, dx, dy, color='red', head_width=5, head_length=5, alpha=0.7)

    # Add colorbar for density
    cbar = plt.colorbar(sc)
    cbar.set_label("Local Density")

    plt.title(f"Frame {graph.frame_id} - Nodes: {graph.x.shape[0]} | Edges: {graph.edge_index.shape[1]}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved visualization → {save_path}")
    else:
        plt.show()


def make_video(image_folder, output_path, fps=5):
    """Combine all .png images in folder into a video."""
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if not images:
        print("[ERROR] No PNG images found for video export.")
        return

    # Get frame size from first image
    first_img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in images:
        frame = cv2.imread(os.path.join(image_folder, img_file))
        video.write(frame)

    video.release()
    print(f"[OK] Video saved → {output_path}")


if __name__ == "__main__":
    # Path setup
    seq_id = "00001"
    seq_path = f"D:/stgnn_project/data/processed/txt_graphs/train/{seq_id}.pt"
    out_dir = f"D:/stgnn_project/outputs/visualizations/{seq_id}"
    os.makedirs(out_dir, exist_ok=True)

    graphs = torch.load(seq_path, weights_only=False)

    # Save all frame visualizations
    for g in graphs:
        save_path = os.path.join(out_dir, f"{seq_id}_frame{g.frame_id:03d}.png")
        visualize_frame(g, save_path=save_path)

    # Make video
    video_path = os.path.join(out_dir, f"{seq_id}_viz.mp4")
    make_video(out_dir, video_path, fps=5)  # adjust fps (frames/sec) as you like