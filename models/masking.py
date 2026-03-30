import torch


def mask_adj_matrices_edges(batch_adj_matrices: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
    """Randomly masks off-diagonal graph edges in a batch of adjacency matrices."""
    batch_size, num_nodes, _ = batch_adj_matrices.shape

    all_indices = torch.stack(torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes)), dim=-1).reshape(-1, 2)
    off_diag_indices = all_indices[all_indices[:, 0] != all_indices[:, 1]]
    total_off_diag = off_diag_indices.shape[0]

    mask_count = int(mask_ratio * total_off_diag)
    random_indices = torch.randperm(total_off_diag, device=batch_adj_matrices.device)[:mask_count]

    mask = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=batch_adj_matrices.device)
    chosen = off_diag_indices.to(batch_adj_matrices.device)[random_indices]
    mask[:, chosen[:, 0], chosen[:, 1]] = False

    return batch_adj_matrices * mask


def temporal_patch_masking(x: torch.Tensor, num_splits: int = 8, num_masked: int = 4):
    """Temporal masking by splitting each channel sequence into patches and zeroing sampled patches."""
    batch_size, num_channels, seq_len = x.shape
    patch_len = seq_len // num_splits
    patches = x.view(batch_size, num_channels, num_splits, patch_len)
    masked_patches = patches.clone()

    rand_indices = torch.rand(x.shape[1], num_splits, device=x.device).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]

    mask = torch.zeros_like(patches, dtype=torch.bool)
    batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).unsqueeze(2)
    mask[
        batch_indices,
        torch.arange(x.shape[1], device=x.device).view(1, -1, 1),
        selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1),
    ] = True

    masked_patches[mask] = 0

    mask = mask.contiguous().view(batch_size, num_channels, -1)
    masked_x = masked_patches.contiguous().view(batch_size, num_channels, -1)
    return masked_x, mask


# Backward-compatible alias used by existing code paths.
def masking2(x, num_splits=8, num_masked=4):
    return temporal_patch_masking(x, num_splits=num_splits, num_masked=num_masked)
