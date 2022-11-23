import os
import torch

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

import part_seg_pathwise.tools.utils as utils

from sklearn.metrics import accuracy_score

from part_seg_pathwise.datasets.part_net_seg import PartNetMultiLevelSegmentation
from part_seg_pathwise.models.net import PartGLOM
from part_seg_pathwise.runners.runner import Runner


class SampleRunner(Runner):
    def __init__(self, args):
        super(SampleRunner, self).__init__(args)

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
        self.print_log(
            f"Test data loaded: {len(self.dataset['test'])} samples."
        )

    def load_model(self):
        model = PartGLOM(level_classes=self.level_classes)
        self.model = model.to(self.output_dev)

    def initialize_model(self):
        if self.args.weights is not None:
            self.load_model_weights(
                self.model,
                self.args.weights,
                self.args.ignore_weights
            )
        else:
            raise ValueError('Please appoint --weights.')

    def run(self):
        if dist.get_rank() == 0:
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
        self.model.eval()

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
            # forward
            out = self.model(
                x,
                normals=normal,
                sampling=self.args.sampling,
                tau=self.args.tau,
                hard=self.args.hard_sampling
            )
            mid_logits = out['mid_score']
            out5 = out['out5']
            embed = out['mid_embeddings']
            mid_pred_probs = []
            top_pred_probs = []
            for s in range(self.args.num_samples):
                current_probs = F.gumbel_softmax(
                    mid_logits, tau=self.args.tau,
                    hard=self.args.hard_sampling, dim=-1
                )
                mid_pred_probs.append(current_probs)
                top_out = self.model.module.sample_inference(x, out5, embed, current_probs)
                top_pred = F.softmax(top_out['top_score'], dim=-1)
                top_pred_probs.append(top_pred)
            mid_pred = torch.stack(mid_pred_probs)
            top_pred = torch.stack(top_pred_probs)
            assert top_pred.size(0) == self.args.num_samples and \
                   mid_pred.size(0) == self.args.num_samples
            mid_pred = torch.mean(mid_pred, dim=0)
            top_pred = torch.mean(top_pred, dim=0)

            # gather prediction and ground-truth
            pred_scores['mid'].append(mid_pred.max(dim=1)[1])
            pred_scores['top'].append(top_pred.max(dim=1)[1])
            for i, level in enumerate(['mid', 'top']):
                true_label = label[:, i, :].reshape(-1)
                true_scores[level].append(true_label)

            if self.has_dummy_class[-1]:
                valid = true_scores['top'][-1] != self.level_classes[-1] - 1
            else:
                valid = torch.ones_like(true_scores['top'][-1], dtype=torch.bool)
            loss = F.cross_entropy(
                top_pred[valid], true_scores['top'][-1][valid]
            )
            # loss = self.distributed_reduce_mean(loss)
            loss_values.append(loss.item())
            # store necessary data
            point_clouds.append(x.cpu())  # (B, N, 3)
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['test']), loss.item(),
                        self.optimizer.param_groups[0]['lr']
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
            if self.has_dummy_class[0]:
                acc_without[level] = utils.clustering_acc(
                    true_scores[level], pred_scores[level],
                    self.args.num_points, self.level_classes[0] - 1
                )['accuracy']
            else:
                acc_without[level] = acc_with[level]
            # update the score
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

        with open('./sampled_results_gumbel_{}.txt'.format(self.args.category), 'a+') as f:
            print('{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
                self.args.num_samples, self.args.category,
                acc_with['mid'] * 100.0, acc_with['top'] * 100.0,
                acc_without['mid'] * 100.0, acc_with['top'] * 100.0,
                iou_with['mid'] * 100.0, iou_with['top'] * 100.0
            ), file=f)
