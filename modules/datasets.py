import os

import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial


def identity(x):
    return x


def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f % frames == 0:
        return t[:, :-(frames - 1)]
    if f % frames == 1:
        return t
    else:
        return t[:, :-((f % frames) - 1)]


class BaseDataset(Dataset):
    def __init__(self, args, split, tokenizer, min_slices=20, resize_dim=500, num_frames=2, force_num_frames=True):
        self.image_dir = args.image_dir
        self.ann_file = args.ann_file
        self.tokenizer = tokenizer
        self.min_slices = min_slices
        self.image_size = args.image_size
        self.voxel_depth = args.voxel_depth
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.clip_top_k = args.clip_top_k
        self.ann = json.loads(open(self.ann_file, 'r').read())
        self.samples = self.ann[self.split]
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor()
        ])
        self.clip_report_embeddings = np.load(args.clip_report_embedding_path)['data']
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform=self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity
        for i in range(len(self.samples)):
            self.samples[i]['ids'] = tokenizer(self.samples[i]['findings'])[:self.max_seq_length]
            self.samples[i]['mask'] = [1] * len(self.samples[i]['ids'])

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        img_data = np.load(path)["arr_0"]  # (depth, height, width)

        # Generate tensors for three different views
        tensor_axial = self.process_img_data(np.transpose(img_data, (1, 2, 0)))  # (height, width, depth)
        tensor_sagittal = self.process_img_data(np.transpose(img_data, (0, 2, 1)))  # (depth, width, height)
        tensor_coronal = self.process_img_data(np.transpose(img_data, (0, 1, 2)))  # (depth, height, width)

        return tensor_axial, tensor_sagittal, tensor_coronal

    def process_img_data(self, img_data):
        img_data = img_data * 1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data + 400) / 600)).astype(np.float32)

        tensor = torch.tensor(img_data)

        # Get the dimensions of the input tensor
        target_shape = (self.image_size, self.image_size, self.voxel_depth)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (
            pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]


class CTReportDataset(BaseDataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        clip_indices = sample['clip_indices'][:self.clip_top_k]
        clip_memory = self.clip_report_embeddings[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()
        image_id = sample['id']
        image_path = os.path.join(self.image_dir, sample['image_path'][0])
        tensor_axial, tensor_sagittal, tensor_coronal = self.nii_to_tensor(image_path)
        finding_ids = sample['ids']
        finding_masks = sample['mask']
        seq_length = len(finding_ids)
        example = (image_id, clip_memory, tensor_axial, tensor_sagittal,
                   tensor_coronal, finding_ids, finding_masks, seq_length)
        return example
