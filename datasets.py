from typing import List
import os.path as osp
import torch
from utils.mimic_dataset import read_data

from torch_geometric.data import InMemoryDataset


class MIMICDataset(InMemoryDataset):

    def __init__(self, root: str, transform=None, pre_transform=None):
        self.name = 'MIMICDataset'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = [
            'admissions', 'services', 'icustays'
        ]
        return [f'{name}.csv' for name in names]

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        data_list = read_data(self.raw_dir, self.processed_dir)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return self.name+'()'
