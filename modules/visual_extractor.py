import torch
import torch.nn as nn


class VisualExtractor(nn.Module):
    def __init__(self, feature_model, args):
        super(VisualExtractor, self).__init__()
        self.image_size = args.image_size
        self.spatial_patch_size = args.spatial_patch_size
        self.model = feature_model
        self.kernel_size = int(self.image_size / self.spatial_patch_size)
        assert self.image_size % self.spatial_patch_size == 0, "image size must be divisible by spatial patch size"
        self.avg_fnt = torch.nn.AvgPool3d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images, return_encoded_tokens=True)
        patch_feats = patch_feats.permute(0, 4, 1, 2, 3)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
