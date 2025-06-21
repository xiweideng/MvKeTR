import torch
import numpy as np
from torch.utils.data import DataLoader

from .datasets import CTReportDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.dataset = CTReportDataset(self.args, split=self.split, tokenizer=self.tokenizer)
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, clip_memory, images_axial, images_sagittal, images_coronal, reports_ids, reports_masks, seq_lengths = zip(
            *data)
        clip_memory = torch.stack(clip_memory, 0)
        images_axial = torch.stack(images_axial, 0)
        images_sagittal = torch.stack(images_sagittal, 0)
        images_coronal = torch.stack(images_coronal, 0)
        max_seq_length = max(seq_lengths)
        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return (images_id, clip_memory, images_axial, images_sagittal, images_coronal,
                torch.LongTensor(targets), torch.FloatTensor(targets_masks))
