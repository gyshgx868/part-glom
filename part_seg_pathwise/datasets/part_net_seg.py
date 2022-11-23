import h5py
import os
import pickle

import numpy as np
import torch.utils.data as data

import part_seg_pathwise.tools.utils as utils


class PartNetMultiLevelSegmentation(data.Dataset):
    def __init__(self, data_path, category, phase, num_points=2048):
        super(PartNetMultiLevelSegmentation, self).__init__()
        self.data_path = os.path.join(data_path, 'h5_files')
        self.phase = phase
        self.category = category
        self.num_points = num_points

        data_file = os.path.join(self.data_path, f'{category}.{phase}.h5')
        with h5py.File(data_file, 'r') as f:
            self.points = f['points'][:]
            self.labels = f['labels'][:]
        info_file = os.path.join(self.data_path, f'{category}.info')
        with open(info_file, 'rb') as f:
            info_data = pickle.load(f)
            self.class_names = info_data['class_names']
            self.hierarchy_colors = info_data['hierarchy_colors']

        dummy_mid_class = 1 if np.any(self.labels[:, :, 0] == -1) else 0
        dummy_top_class = 1 if np.any(self.labels[:, :, 1] == -1) else 0
        self.has_dummy_class = [bool(dummy_mid_class), bool(dummy_top_class)]

        # remove samples with all -1 labeled
        valid = []
        for i in range(self.labels.shape[0]):
            mask = self.labels[i, :, -1]
            assert len(mask.shape) == 1 and mask.shape[0] == 2048
            valid.append(int(np.sum(mask) != -2048))
        valid = np.array(valid, dtype=bool)
        self.points, self.labels = self.points[valid], self.labels[valid]

        self.level_classes = [
            len(self.class_names['mid']) + dummy_mid_class,
            len(self.class_names['top']) + dummy_top_class
        ]
        for i in range(len(self.level_classes)):
            self.labels[self.labels[:, :, i] == -1, i] = \
                self.level_classes[i] - 1
        self.labels = np.transpose(self.labels, axes=(0, 2, 1))

        self.weights = {
            'mid': self._get_weights(self.labels[:, 0, :]),
            'top': self._get_weights(self.labels[:, -1, :])
        }

        rgb_colors = {
            'mid': np.zeros(shape=(self.level_classes[0], 3), dtype=float),
            'top': np.zeros(shape=(self.level_classes[1], 3), dtype=float),
        }
        for level in ['mid', 'top']:
            for k, v in self.hierarchy_colors[level].items():
                rgb_colors[level][k, :] = np.array(v)
        self.hierarchy_colors = rgb_colors

        normal_file = os.path.join(
            self.data_path, f'{category}.{phase}.norm.h5'
        )
        if os.path.exists(normal_file):
            with h5py.File(normal_file, 'r') as f:
                self.normals = f['normals'][:]
        else:
            # self.normals = self.points
            print('Normal estimating: k = 20')
            self.normals = []
            from tqdm import tqdm
            for points in tqdm(self.points):
                normal = utils.compute_normals(points, k=20)
                self.normals.append(normal)
            self.normals = np.stack(self.normals)
            with h5py.File(normal_file, 'w') as f:
                f.create_dataset(
                    'normals', data=self.normals,
                    compression='gzip', compression_opts=4, dtype='float32'
                )
            print(self.normals.shape)
            print('successfully written!')

    def _get_weights(self, level_label):
        label = level_label.reshape(-1)
        _, counts = np.unique(label, return_counts=True)
        return 1.0 / counts

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        x = self.points[index, 0:self.num_points, :]
        label = self.labels[index, :, 0:self.num_points]
        normal = self.normals[index, 0:self.num_points, :]
        return {
            'points': x,  # (N, 3)
            'labels': label,  # (2, N)
            'normals': normal,  # (N, 3)
        }


def main():
    dataset = PartNetMultiLevelSegmentation(
        '../../data', category='Table', phase='train'
    )
    data_len = len(dataset)
    print('#Samples:', data_len)
    print('#Classes:', dataset.level_classes)
    print(dataset.hierarchy_colors)
    for i in range(data_len):
        data = dataset.__getitem__(i)
        x = data['points']
        y = data['labels']
        n = data['normals']
        print(x.shape, y.shape, n.shape)


if __name__ == '__main__':
    main()
