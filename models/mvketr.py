import numpy as np
import torch
import torch.nn as nn
from ctvit import CTViT

from modules.base_cmn import BaseCMN
from modules.cross_modal_knowledge_enhancer import CrossModalKnowledgeEnhancer
from modules.visual_extractor import VisualExtractor
from modules.multi_view_perception_aggregator import MultiviewPerceptionAggregator


class MvKeTR(nn.Module):
    def __init__(self, args, tokenizer):
        super(MvKeTR, self).__init__()
        self.args = args
        self.spatial_patch_size = args.spatial_patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.image_size = args.image_size
        self.d_vf = args.d_vf
        self.cross_modal_knowledge_enhancer = CrossModalKnowledgeEnhancer(d_model=512)
        self.tokenizer = tokenizer
        coro_model = CTViT(
            dim=self.d_vf,
            codebook_size=8192,
            image_size=self.image_size,
            patch_size=self.spatial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        axi_model = CTViT(
            dim=self.d_vf,
            codebook_size=8192,
            image_size=self.image_size,
            patch_size=self.spatial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        sag_model = CTViT(
            dim=self.d_vf,
            codebook_size=8192,
            image_size=self.image_size,
            patch_size=self.spatial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        self.coronal_visual_extractor = VisualExtractor(coro_model, args)
        self.axial_visual_extractor = VisualExtractor(axi_model, args)
        self.sagittal_visual_extractor = VisualExtractor(sag_model, args)
        self.multi_view_perception_aggregator = MultiviewPerceptionAggregator(self.d_vf)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        self.forward = self.forward_ct2rep

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_ct2rep(self, clip_memory, images_axial, images_sagittal, images_coronal, targets=None, mode='train'):
        att_feats_axial, fc_feats_axial = self.axial_visual_extractor(images_axial)
        cross_modal_informed_token = self.cross_modal_knowledge_enhancer(att_feats_axial, clip_memory)
        att_feats_sagittal, fc_feats_sagittal = self.sagittal_visual_extractor(images_sagittal)
        att_feats_coronal, fc_feats_coronal = self.coronal_visual_extractor(images_coronal)
        fc_feats_axial = fc_feats_axial.unsqueeze(1)
        fc_feats_sagittal = fc_feats_sagittal.unsqueeze(1)
        fc_feats_coronal = fc_feats_coronal.unsqueeze(1)
        fused_att_feats = self.multi_view_perception_aggregator(att_feats_axial, att_feats_sagittal, att_feats_coronal)
        fused_fc_feats = self.multi_view_perception_aggregator(fc_feats_axial, fc_feats_sagittal, fc_feats_coronal)
        fused_cat_att_feats = torch.cat([fused_att_feats, cross_modal_informed_token], dim=1)
        fused_fc_feats = fused_fc_feats.squeeze(1)
        if mode == 'train':
            output = self.encoder_decoder(fused_fc_feats, fused_cat_att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fused_fc_feats, fused_cat_att_feats, mode='sample')
        else:
            raise ValueError
        return output
