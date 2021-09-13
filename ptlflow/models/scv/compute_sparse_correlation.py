import torch
from .knn import knn_faiss_raw
from .utils import coords_grid, coords_grid_y_first


def normalize_coords(coords, H, W):
    """ Normalize coordinates based on feature map shape. coords: [B, 2, N]"""
    one = coords.new_tensor(1)
    size = torch.stack([one*W, one*H])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.5
    return (coords - center[:, :, None]) / scaling[:, :, None]


def compute_sparse_corr_init(fmap1, fmap2, k=32):
    """
    Compute a cost volume containing the k-largest hypotheses for each pixel.
    Output: corr_mink
    """
    B, C, H1, W1 = fmap1.shape
    H2, W2 = fmap2.shape[2:]
    N = H1 * W1

    fmap1, fmap2 = fmap1.view(B, C, -1), fmap2.view(B, C, -1)

    with torch.no_grad():
        _, indices = knn_faiss_raw(fmap1, fmap2, k)  # [B, k, H1*W1]

        indices_coord = indices.unsqueeze(1).expand(-1, 2, -1, -1)  # [B, 2, k, H1*W1]
        coords0 = coords_grid_y_first(B, H2, W2).view(B, 2, 1, -1).expand(-1, -1, k, -1).to(fmap1.device)  # [B, 2, k, H1*W1]
        coords1 = coords0.gather(3, indices_coord)  # [B, 2, k, H1*W1]
        coords1 = coords1 - coords0

        # Append batch index
        batch_index = torch.arange(B).view(B, 1, 1, 1).expand(-1, -1, k, N).type_as(coords1)

    # Gather by indices from map2 and compute correlation volume
    fmap2 = fmap2.gather(2, indices.view(B, 1, -1).expand(-1, C, -1)).view(B, C, k, N)
    me_corr = torch.einsum('bcn,bckn->bkn', fmap1, fmap2).contiguous() / torch.sqrt(torch.tensor(C).float())  # [B, k, H1*W1]

    return me_corr, coords0, coords1, batch_index  # coords: [B, 2, k, H1*W1]


if __name__ == "__main__":
    torch.manual_seed(0)

    for _ in range(100):
        fmap1 = torch.randn(8, 256, 92, 124).cuda()
        fmap2 = torch.randn(8, 256, 92, 124).cuda()
        corr_me = compute_sparse_corr_init(fmap1, fmap2, k=16)

    # corr_dense = corr(fmap1, fmap2)
    # corr_max = torch.max(corr_dense, dim=3)