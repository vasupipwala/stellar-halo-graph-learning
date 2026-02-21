"""
stream_gnn.py

Graph Neural Network pipeline
for stellar stream representation learning.

Author: Vasu Pipwala
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import pairwise_distances
from typing import List


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_stream_graph(stream_snapshot, halo_id, k=k, max_nodes=1500, device="cpu"):
    """
    Build a stream graph from a phase-space snapshot.

    Node features:
        (x, y, z, vx, vy, vz)

    Edge features:
        scalar distance (rotation invariant)

    Parameters
    ----------
    stream_snapshot : gala PhaseSpacePosition
    halo_id : int
    k : int
    max_nodes : int
    device : str
    """

    # -----------------------------------
    # Extract phase-space
    # -----------------------------------
    pos = stream_snapshot.pos.xyz.T.to_value()
    vel = stream_snapshot.vel.d_xyz.T.to_value()

    pos = torch.tensor(pos, dtype=torch.float, device=device)
    vel = torch.tensor(vel, dtype=torch.float, device=device)

    # -----------------------------------
    # Subsample BEFORE graph construction
    # -----------------------------------
    N = pos.size(0)
    if N > max_nodes:
        idx = torch.randperm(N, device=device)[:max_nodes]
        pos = pos[idx]
        vel = vel[idx]

    # -----------------------------------
    # Center positions (translation invariance)
    # -----------------------------------
    pos = pos - pos.mean(dim=0, keepdim=True)

    # -----------------------------------
    # Velocity normalization (per-stream)
    # -----------------------------------
    vel_std = vel.std(dim=0, keepdim=True) + 1e-8
    vel = vel / vel_std

    # -----------------------------------
    # Node features: pure 6D phase-space
    # -----------------------------------
    x = torch.cat([pos, vel], dim=1)   # (N,6)

    # -----------------------------------
    # Build kNN graph
    # -----------------------------------
    N = pos.size(0)
    k = min(k, N - 1)

    with torch.no_grad():
        dist = torch.cdist(pos, pos)
        knn_idx = dist.topk(k=k+1, largest=False).indices[:, 1:]

    row = torch.arange(N, device=device).unsqueeze(1).repeat(1, k).flatten()
    col = knn_idx.flatten()

    edge_index = torch.stack([row, col], dim=0)

    # Edge feature: scalar distance only (rotation invariant)
    edge_attr = dist[row, col].unsqueeze(1)

    graph = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        halo_id=torch.tensor([halo_id], device=device)
    )

    return graph


def random_small_rotation(device="cpu", max_angle=0.2):
    """
    Small random rotation around random axis.
    max_angle in radians (~0.3 ≈ 17 degrees)
    """
    axis = torch.randn(3, device=device)
    axis = axis / torch.norm(axis)

    angle = torch.rand(1, device=device) * max_angle

    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device)

    R = torch.eye(3, device=device) + torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)

    return R


def add_position_noise(pos, noise_fraction=0.005):
    """
    Add small isotropic Gaussian noise scaled to RMS radius.
    """

    rms = torch.sqrt(torch.mean(torch.sum(pos**2, dim=1)))
    scale = noise_fraction * rms

    noise = scale * torch.randn_like(pos)

    return pos + noise


def build_knn_graph(pos, vel, halo_id, k=k):
    """
    Build kNN graph with 6D node features:
        (x, y, z, vx, vy, vz)
    """

    device = pos.device
    N = pos.size(0)
    k = min(k, N - 1)

    # -----------------------------------
    # Node features: (x,y,z,vx,vy,vz)
    # -----------------------------------
    x = torch.cat([pos, vel], dim=1)   # (N,6)

    # -----------------------------------
    # kNN in position space
    # -----------------------------------
    with torch.no_grad():
        dist = torch.cdist(pos, pos)
        knn_idx = dist.topk(k=k+1, largest=False).indices[:, 1:]

    row = torch.arange(N, device=device).unsqueeze(1).repeat(1, k).flatten()
    col = knn_idx.flatten()

    edge_index = torch.stack([row, col], dim=0)

    # Edge features: scalar distance (rotation invariant)
    edge_attr = dist[row, col].unsqueeze(1)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        halo_id=halo_id.clone()
    )

def subsample_graph(data, keep_ratio=0.9, k=k):
    """
    Subsample nodes and rebuild graph consistently (6D setup).
    """

    device = data.x.device
    N = data.x.size(0)

    if k is None:
        k = 8

    keep_N = max(16, int(N * keep_ratio))
    keep_N = min(keep_N, N)
    k = min(k, keep_N - 1)

    idx = torch.randperm(N, device=device)[:keep_N]

    pos = data.pos[idx]
    vel = data.x[idx][:, 3:6]   # since x = (x,y,z,vx,vy,vz)

    return build_knn_graph(
        pos=pos,
        vel=vel,
        halo_id=data.halo_id,
        k=k
    )

def augment_graph(data, k=k, keep_ratio=0.9):
    """
    Physics-preserving augmentation:
        - SO(2) rotation
        - Small isotropic position noise
        - Optional subsampling
    """

    device = data.pos.device

    if k is None:
        k = 8

    # -----------------------------------
    # Extract tensors
    # -----------------------------------
    pos = data.pos.clone()
    vel = data.x[:, 3:6].clone()

    # -----------------------------------
    # 1️⃣ Random rotation (SO(2))
    # -----------------------------------
    R = random_small_rotation(device=device, max_angle=0.2)

    pos = pos @ R.T
    vel = vel @ R.T

    # -----------------------------------
    # 2️⃣ Small isotropic noise
    # -----------------------------------
    pos = add_position_noise(pos, noise_fraction=0.003)

    # -----------------------------------
    # 3️⃣ Rebuild graph (NO r anymore)
    # -----------------------------------
    augmented = build_knn_graph(
        pos=pos,
        vel=vel,
        halo_id=data.halo_id,
        k=k
    )

    # -----------------------------------
    # 4️⃣ Optional subsampling
    # -----------------------------------
    if keep_ratio < 1.0:
        augmented = subsample_graph(
            augmented,
            keep_ratio=keep_ratio,
            k=k
        )

    return augmented


class EGNNLayer(nn.Module):

    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index):

        row, col = edge_index

        # Rotation-invariant squared distance
        rel_pos = pos[row] - pos[col]
        dist2 = torch.sum(rel_pos ** 2, dim=1, keepdim=True)

        # Edge message
        edge_input = torch.cat([x[row], x[col], dist2], dim=1)
        edge_feat = self.edge_mlp(edge_input)

        # Aggregate messages
        agg = torch.zeros(
            x.size(0),
            x.size(1),
            device=x.device
        )
        agg.index_add_(0, row, edge_feat)

        # Node update
        update = self.node_mlp(torch.cat([x, agg], dim=1))
        update = self.dropout(update)

        # Residual + normalization
        x = self.norm(x + update)

        return x


class EGNNEncoder(nn.Module):

    def __init__(self, in_features=6, hidden=hidden, layers=3, emb_dim=24, dropout=0.05):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList([
            EGNNLayer(hidden, dropout=dropout)
            for _ in range(layers)
        ])

        self.lin = nn.Linear(hidden, emb_dim)

    def forward(self, data):

        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch

        # Project to hidden space
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, pos, edge_index)

        # Invariant graph pooling
        graph_embedding = global_mean_pool(x, batch)

        z = self.lin(graph_embedding)

        return z


def halo_prototype_loss(z, labels, temperature=0.07):
    """
    Simple, stable halo clustering loss.

    Steps:
    1. Normalize embeddings
    2. Compute halo centroids
    3. Classify each embedding via distance to centroids
    """

    # Normalize
    z = F.normalize(z, dim=1)

    unique_labels = labels.unique()
    centroids = []

    for h in unique_labels:
        mask = labels == h
        centroids.append(z[mask].mean(dim=0))

    centroids = torch.stack(centroids)  # (H, d)
    centroids = F.normalize(centroids, dim=1)

    # Compute cosine similarity to centroids
    logits = torch.matmul(z, centroids.T) / temperature

    # Map labels to 0..H-1 index
    label_map = {int(h.item()): i for i, h in enumerate(unique_labels)}
    target = torch.tensor(
        [label_map[int(l.item())] for l in labels],
        device=z.device
    )

    loss = F.cross_entropy(logits, target)

    return loss


def multi_positive_supervised_contrastive_loss(
    z,
    halo_labels,
    tau=0.12
):
    """
    Multi-positive supervised contrastive loss
    (Khosla et al., 2020)

    Positives:
        - All samples belonging to same halo
        - Includes same stream across time automatically

    Parameters
    ----------
    z : Tensor (N, d)
        Graph embeddings
    halo_labels : Tensor (N,)
        Halo ID for each graph
    tau : float
        Temperature parameter

    Returns
    -------
    loss : scalar Tensor
    """

    # ----------------------------------------
    # Normalize embeddings (critical)
    # ----------------------------------------
    z = F.normalize(z, dim=1)

    N = z.size(0)

    # ----------------------------------------
    # Similarity matrix
    # ----------------------------------------
    sim = torch.matmul(z, z.T) / tau

    # Mask self-similarity
    self_mask = torch.eye(N, device=z.device).bool()
    sim = sim.masked_fill(self_mask, -1e9)

    # ----------------------------------------
    # Build positive mask (same halo)
    # ----------------------------------------
    halo_labels = halo_labels.view(-1, 1)

    positive_mask = (halo_labels == halo_labels.T)
    positive_mask = positive_mask & (~self_mask)

    # ----------------------------------------
    # Log-softmax denominator
    # ----------------------------------------
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # ----------------------------------------
    # Count positives per anchor
    # ----------------------------------------
    pos_count = positive_mask.sum(dim=1)

    # Remove anchors without positives (should not happen)
    valid = pos_count > 0

    # ----------------------------------------
    # Final loss
    # ----------------------------------------
    loss = -(
        (positive_mask * log_prob).sum(dim=1)[valid]
        / pos_count[valid]
    ).mean()

    return loss


def supervised_contrastive_loss(z, labels, tau=0.2):
    """
    Standard Supervised Contrastive Loss
    (Khosla et al., 2020)

    Encourages embeddings of streams from the same halo
    to cluster, while separating different halos.

    Parameters
    ----------
    z : (N, d)
        Graph embeddings
    labels : (N,)
        Halo labels
    tau : float
        Temperature
    """

    z = F.normalize(z, dim=1)
    N = z.size(0)

    sim = torch.matmul(z, z.T) / tau

    # Remove self-similarity
    self_mask = torch.eye(N, device=z.device).bool()
    sim = sim.masked_fill(self_mask, -1e9)

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & (~self_mask)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_count = positive_mask.sum(dim=1)
    valid = pos_count > 0

    loss = - (positive_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]

    return loss.mean()


def generate_time_contrastive_batch(
    stream_metadata,
    k=k,
    device="cpu",
    time_samples_per_stream=4
):
    """
    Build a shuffled batch dynamically each epoch.

    - Shuffles stream order
    - Randomly samples time steps
    - Shuffles final graph list before batching
    """

    graphs = []

    # -----------------------------------------
    # 1️⃣ Shuffle stream order
    # -----------------------------------------
    shuffled_streams = np.random.permutation(stream_metadata)

    for meta in shuffled_streams:

        stream = meta["stream_object"]
        time_array = meta["time_array"]
        halo_id = meta["halo_id"]

        total_time_steps = len(time_array)

        # -----------------------------------------
        # 2️⃣ Random time sampling
        # -----------------------------------------
        time_indices = np.random.choice(
            total_time_steps,
            size=time_samples_per_stream,
            replace=False
        )

        for t_idx in time_indices:

            snapshot = extract_stream_snapshot(
                stream,
                time_array,
                time_index=int(t_idx)
            )

            graph = build_stream_graph(
                snapshot,
                halo_id=halo_id,
                k=k,
                device=device
            )

            graphs.append(graph)

    # -----------------------------------------
    # 3️⃣ Shuffle graph order
    # -----------------------------------------
    np.random.shuffle(graphs)

    batch = Batch.from_data_list(graphs).to(device)

    return batch

def build_evaluation_graphs(
    stream_metadata,
    n_eval_times=4,
    k=8,
    device="cpu"
):
    """
    Deterministic evaluation set.
    No augmentation.
    Fixed time slices.
    """

    evaluation_graphs = []

    for meta in stream_metadata:

        stream = meta["stream_object"]
        time_array = meta["time_array"]
        halo_id = meta["halo_id"]

        total_steps = len(time_array)

        eval_indices = np.linspace(
            0,
            total_steps - 1,
            n_eval_times
        ).astype(int)

        for t_idx in eval_indices:

            snapshot = extract_stream_snapshot(
                stream,
                time_array,
                time_index=int(t_idx)
            )

            graph = build_stream_graph(
                snapshot,
                halo_id=halo_id,
                k=k,
                device=device
            )

            evaluation_graphs.append(graph)

    return evaluation_graphs


def compute_halo_distance_ratio(model, evaluation_graph_list, device):

    model.eval()

    with torch.no_grad():
        batch = Batch.from_data_list(evaluation_graph_list).to(device)
        z = model(batch)
        z = F.normalize(z, dim=1)

    embeddings = z.cpu().numpy()
    labels = batch.halo_id.view(-1).cpu().numpy()

    # Pairwise Euclidean distances
    D = pairwise_distances(embeddings, metric="euclidean")

    intra_distances = []
    inter_distances = []

    N = len(labels)

    for i in range(N):
        for j in range(i + 1, N):

            if labels[i] == labels[j]:
                intra_distances.append(D[i, j])
            else:
                inter_distances.append(D[i, j])

    intra_mean = np.mean(intra_distances)
    inter_mean = np.mean(inter_distances)

    ratio = inter_mean / (intra_mean + 1e-8)

    return intra_mean, inter_mean, ratio


