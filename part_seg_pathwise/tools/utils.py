import argparse
import colorsys
import random
import torch

import numpy as np

from torch_geometric.utils import intersection_and_union


def collate_feats_with_none(batch_data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    collated_data = dict()
    for data in batch_data:
        if data is None:
            continue
        for k, v in data.items():
            if k not in collated_data.keys():
                collated_data.update({k: v})
            else:
                collated_data[k] = torch.cat((collated_data[k], v), dim=0)
    return collated_data


def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def generate_hls_colors(num_colors):
    hls_colors = []
    sector = 0
    step = 360.0 / num_colors
    while sector < 360:
        h = sector
        l = 50 + random.random() * 10
        s = 90 + random.random() * 10
        color = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(color)
        sector += step
    hls_colors = np.array(hls_colors, dtype=float)
    return hls_colors


def generate_rgb_colors(num_colors):
    rgb_colors = []
    hls_colors = generate_hls_colors(num_colors)
    for i in range(num_colors):
        h, l, s = hls_colors[i, :]
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        rgb_colors.append([int(r * 255), int(g * 255), int(b * 255)])
    rgb_colors = np.array(rgb_colors, dtype=int)
    return rgb_colors


def hex_to_rgb(hex_color):
    assert hex_color[0] == '#'
    hex_color = hex_color.lower()
    r = int('0x' + hex_color[1:3], 16)
    g = int('0x' + hex_color[3:5], 16)
    b = int('0x' + hex_color[5:7], 16)
    return r, g, b


def clustering_acc(y_true, y_pred, num_points, dummy_class_id=0):
    from sklearn.metrics import accuracy_score
    from part_seg_pathwise.tools.linear_assignment_ import linear_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.shape == y_true.shape
    assert y_pred.shape[0] % num_points == 0
    matched_pred = []
    for point_id in range(0, y_pred.shape[0], num_points):
        cur_true = y_true[point_id:point_id + num_points]
        cur_pred = y_pred[point_id:point_id + num_points]
        if dummy_class_id > 0:
            mask = cur_true != dummy_class_id
            cur_true = cur_true[mask]
            cur_pred = cur_pred[mask]
        D = max(np.max(cur_pred), np.max(cur_true)) + 1
        cost = np.zeros((D, D), dtype=np.int64)
        for i in range(cur_pred.size):
            cost[cur_pred[i], cur_true[i]] += 1
        matched_id = linear_assignment(cost.max() - cost)
        new_pred = -cur_pred
        for i, j in matched_id:
            new_pred[new_pred == -i] = j
        matched_pred.append(new_pred)
    matched_pred = np.concatenate(matched_pred)
    if dummy_class_id > 0:
        y_true = y_true[y_true != dummy_class_id]
    assert matched_pred.shape == y_true.shape
    acc = accuracy_score(y_true, matched_pred)
    clustering_result = {
        'matched_pred': matched_pred, 'accuracy': acc
    }
    return clustering_result


def export_ply_with_color(out, points, colors):
    with open(out, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex ' + str(points.shape[0]) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            if isinstance(colors[i, 0], float):
                color = [
                    int(colors[i, 0] * 255), int(colors[i, 1] * 255),
                    int(colors[i, 2] * 255),
                ]
            else:
                color = colors[i, :]
            f.write('%f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2],
                color[0], color[1], color[2]
            ))


def compute_normals(points, k=20):
    num_points = points.shape[0]
    t = np.sum(points**2, axis=1)
    A = np.tile(t, (num_points, 1))
    C = np.array(A).T
    B = np.dot(points, points.T)
    dist = A - 2 * B + C
    neigh_ids = dist.argsort(axis=1)[:, :k]
    vec_ones = np.ones((k, 1)).astype(np.float32)
    normals = np.zeros((num_points, 3)).astype(np.float32)
    for idx in range(num_points):
        D = points[neigh_ids[idx, :], :]
        cur_normal = np.dot(np.linalg.pinv(D), vec_ones)
        cur_normal = np.squeeze(cur_normal)
        len_normal = np.sqrt(np.sum(cur_normal ** 2))
        normals[idx, :] = cur_normal / len_normal
        if np.dot(normals[idx, :], points[idx, :]) < 0:
            normals[idx, :] = -normals[idx, :]
    return normals


def mean_iou(pred, target, num_classes, batch=None, omitnans=False):
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: :class:`Tensor`
    """
    pred = torch.LongTensor(pred)
    target = torch.LongTensor(target)
    i, u = intersection_and_union(pred, target, num_classes, batch)
    iou = i.to(torch.float) / u.to(torch.float)
    if omitnans:
        iou = iou[~iou.isnan()].mean()
    else:
        iou[torch.isnan(iou)] = 1
        iou = iou.mean()
    return iou.item()
