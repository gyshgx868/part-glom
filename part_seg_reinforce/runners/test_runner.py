import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import part_seg_reinforce.tools.utils as utils

from sklearn.metrics import accuracy_score

from part_seg_reinforce.datasets.part_net_seg import PartNetMultiLevelSegmentation
from part_seg_reinforce.models.net import MiddleModel
from part_seg_reinforce.models.net import TopModel
from part_seg_reinforce.runners.runner import Runner


class TestRunner(Runner):
    def __init__(self, args):
        super(TestRunner, self).__init__(args)

    def load_dataset(self):
        self.dataset['test'] = PartNetMultiLevelSegmentation(
            data_path=self.args.data_path,
            category=self.args.category,
            phase='test',
            num_points=self.args.num_points
        )
        self.level_classes = self.dataset['test'].level_classes
        self.has_dummy_class = self.dataset['test'].has_dummy_class
        self.level_colors = self.dataset['test'].hierarchy_colors
        self.class_names = {
            'mid': list(self.dataset['test'].class_names['mid'].keys())
        }
        if self.has_dummy_class[0]:
            self.class_names['mid'].append('dummy')
        self.print_log(f"Test data loaded: {len(self.dataset['test'])} samples.")

    def load_model(self):
        self.model1 = MiddleModel(self.level_classes).to(self.output_dev)
        self.model2 = TopModel(self.level_classes).to(self.output_dev)

    def initialize_model(self):
        if self.args.weights is not None:
            self.load_weights(self.args.weights, self.args.ignore_weights)
        else:
            raise ValueError('Please appoint --weights.')

    def run(self):
        self._eval()

    def _save_point_cloud_ply(self, point_clouds, gt_scores, pred_scores):
        to_save = []
        all_labels = []
        for i in range(point_clouds.size(0)):
            x = point_clouds[i].cpu().numpy()
            N = x.shape[0]
            start = i * N
            end = (i + 1) * N
            cur_sample = (x, i)
            cur_score = 0
            pred_colors = {'mid': None, 'top': None}
            gt_colors = {'mid': None, 'top': None}
            for level in ['mid', 'top']:
                pred_labels = pred_scores[level][start:end]
                gt_labels = gt_scores[level][start:end]
                if level != 'top':
                    cur_score += np.sum(gt_labels == pred_labels)
                pred_colors[level] = np.array(self.level_colors[level][pred_labels])
                gt_colors[level] = np.array(self.level_colors[level][gt_labels])
                if level == 'mid':
                    all_labels.append(set(pred_labels))
            cur_sample += (pred_colors, gt_colors, cur_score,)
            to_save.append(cur_sample)
            print(all_labels[-1])
        to_save = sorted(to_save, key=lambda item: item[-1], reverse=True)
        paths = {
            'mid': os.path.join(self.cloud_path, 'mid'),
            'top': os.path.join(self.cloud_path, 'top')
        }
        for k, v in paths.items():
            if not os.path.exists(v):
                os.makedirs(v)
        for i, (x, origin_id, pred_y, gt_y, _) in enumerate(to_save):
            for level in ['mid', 'top']:
                save_file = os.path.join(paths[level], f'{i}_{origin_id}_gt.ply')
                utils.export_ply_with_color(save_file, x, gt_y[level])
                save_file = os.path.join(paths[level], f'{i}_{origin_id}_pred.ply')
                utils.export_ply_with_color(save_file, x, pred_y[level])

    @torch.no_grad()
    def _eval(self):
        self.print_log('Eval Model:')
        self.model1.eval()
        self.model2.eval()

        loss_values = []
        point_clouds = []
        pred_scores = {'top': [], 'mid': []}
        true_scores = {'top': [], 'mid': []}
        acc_with = {'mid': 0.0, 'top': 0.0}
        acc_without = {'mid': 0.0, 'top': 0.0}
        iou_with = {'mid': 0.0, 'top': 0.0}

        timer = 0.0
        self.record_time()
        for batch_id, data in enumerate(self.dataset['test']):
            # data
            x = data['points'].float().to(self.output_dev)
            label = data['labels'].long().to(self.output_dev)
            normal = data['normals'].float().to(self.output_dev)
            # B, N, C = x.size()

            # forward
            out1 = self.model1(x, normals=normal, sampling='categorical')
            mid_pred = out1['logits']
            out2 = self.model2(out1['out5'], out1['embeddings'], out1['probs'], normals=normal)
            top_pred = out2['logits']
            # statistic
            pred_scores['mid'].append(mid_pred.max(dim=1)[1])
            pred_scores['top'].append(top_pred.max(dim=1)[1])
            # gather prediction and ground-truth
            for i, level in enumerate(['mid', 'top']):
                true_label = label[:, i, :].reshape(-1)
                true_scores[level].append(true_label)
            if self.has_dummy_class[-1]:
                valid = true_scores['top'][-1] != self.level_classes[-1] - 1
            else:
                valid = torch.ones_like(true_scores['top'][-1], dtype=torch.bool)
            loss = F.nll_loss(
                out2['log_probs'][valid], true_scores['top'][-1][valid]
            )
            loss_values.append(loss.item())
            # store necessary data
            point_clouds.append(x.cpu())  # (B, N, 3)
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['test']), loss.item(),
                        self.optimizer1.param_groups[0]['lr']
                    ))
            timer += self.tick()

        point_clouds = torch.cat(point_clouds, dim=0)
        # metric calculation
        for level in ['mid', 'top']:
            pred_scores[level] = torch.cat(pred_scores[level], dim=0)
            true_scores[level] = torch.cat(true_scores[level], dim=0)
            pred_scores[level] = pred_scores[level].cpu().numpy()
            true_scores[level] = true_scores[level].cpu().numpy()

        assert true_scores['top'].shape[0] % self.args.num_points == 0
        B = true_scores['top'].shape[0] // self.args.num_points
        batch_for_iou = torch.arange(0, B, dtype=torch.long).reshape(B, 1)
        batch_for_iou = batch_for_iou.repeat(1, self.args.num_points).view(-1)
        assert batch_for_iou.size(0) == true_scores['top'].shape[0]

        acc_with['top'] = accuracy_score(true_scores['top'], pred_scores['top'])
        iou_with['top'] = utils.mean_iou(
            pred_scores['top'], true_scores['top'],
            num_classes=self.level_classes[-1], batch=batch_for_iou
        )
        if self.has_dummy_class[-1]:
            valid = true_scores['top'] != self.level_classes[-1] - 1
            acc_without['top'] = accuracy_score(
                true_scores['top'][valid], pred_scores['top'][valid]
            )
        else:
            acc_without['top'] = acc_with['top']
        for level in ['mid']:
            level_results = utils.clustering_acc(
                true_scores[level], pred_scores[level], self.args.num_points, -1
            )
            acc_with[level] = level_results['accuracy']
            iou_with[level] = utils.mean_iou(
                level_results['matched_pred'], true_scores[level],
                num_classes=self.level_classes[0],
                batch=batch_for_iou
            )
            # ==================================================================
            matched_pred = level_results['matched_pred']
            count = []
            for i in range(0, matched_pred.shape[0], 2048):
                cur_pred = matched_pred[i:i+2048]
                count.append(len(set(cur_pred)))
            pred_bar = np.bincount(count)
            fig = plt.figure(figsize=(15, 15))
            plt.bar(range(len(pred_bar)), pred_bar)
            plt.savefig(os.path.join(self.args.save_dir, 'pred_dist.jpeg'))
            plt.close(fig)
            # ==================================================================
            matched_gt = true_scores[level]
            count = []
            for i in range(0, matched_gt.shape[0], 2048):
                cur_gt = matched_gt[i:i + 2048]
                count.append(len(set(cur_gt)))
            gt_bar = np.bincount(count)
            fig = plt.figure(figsize=(15, 15))
            plt.bar(range(len(gt_bar)), gt_bar)
            plt.savefig(os.path.join(self.args.save_dir, 'gt_dist.jpeg'))
            plt.close(fig)
            # ==================================================================
            if self.has_dummy_class[0]:
                acc_without[level] = utils.clustering_acc(
                    true_scores[level], pred_scores[level],
                    self.args.num_points, self.level_classes[0] - 1
                )['accuracy']
            else:
                acc_without[level] = acc_with[level]
            pred_scores[level] = level_results['matched_pred']

        # metric statistics
        mean_loss = np.mean(loss_values)
        self.print_log('Mean evaluation loss: {:.4f}.'.format(mean_loss))
        self.print_log('Accuracy (w/): {:.2f}% (Mid), {:.2f}% (Top).'.format(
            acc_with['mid'] * 100.0, acc_with['top'] * 100.0
        ))
        self.print_log('Accuracy (w/o): {:.2f}% (Mid), {:.2f}% (Top).'.format(
            acc_without['mid'] * 100.0, acc_with['top'] * 100.0
        ))
        self.print_log('mIoU (w/): {:.2f} (Mid), {:.2f} (Top).'.format(
            iou_with['mid'] * 100.0, iou_with['top'] * 100.0
        ))

        if self.args.save_cloud:
            self.print_log('Saving point clouds...')
            self._save_point_cloud_ply(point_clouds, true_scores, pred_scores)

        self.print_log('Time consumption: {:.1f} min'.format(timer / 60.0))
