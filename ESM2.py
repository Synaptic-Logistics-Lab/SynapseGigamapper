import torch
torch.hub.set_dir("checkpoints/esm2")
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")