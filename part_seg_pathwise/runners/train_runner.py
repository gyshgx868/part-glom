import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

import part_seg_pathwise.tools.utils as utils

from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from part_seg_pathwise.datasets.part_net_seg import PartNetMultiLevelSegmentation
from part_seg_pathwise.models.net import PartGLOM
from part_seg_pathwise.runners.runner import Runner


class TrainRunner(Runner):
    def __init__(self, args):
        super(TrainRunner, self).__init__(args)

    def load_dataset(self):
        self.dataset['train'] = PartNetMultiLevelSegmentation(
            data_path=self.args.data_path,
            category=self.args.category,
            phase='train',
            num_points=self.args.num_points
        )
        self.level_classes = self.dataset['train'].level_classes
        self.has_dummy_class = self.dataset['train'].has_dummy_class
        self.level_colors = self.dataset['train'].hierarchy_colors
        self.class_names = {
            'mid': list(self.dataset['train'].class_names['mid'].keys())
        }
        if self.has_dummy_class[0]:
            self.class_names['mid'].append('dummy')
        self.print_log(f"Train data loaded: {len(self.dataset['train'])} samples.")

        if self.args.eval_model:
            self.dataset['test'] = PartNetMultiLevelSegmentation(
                data_path=self.args.data_path,
                category=self.args.category,
                phase='test',
                num_points=self.args.num_points
            )
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
            self.load_optimizer_weights(self.optimizer, self.args.weights)
            self.load_scheduler_weights(self.scheduler, self.args.weights)

    def run(self):
        # save initialized weights
        model_path = os.path.join(self.model_path, f'model_init.pt')
        self.save_weights(
            -1, self.model, self.optimizer, self.scheduler, model_path
        )

        best_epoch_with = -1
        best_epoch_without = -1
        best_acc_with = [0.0, 0.0]
        best_acc_without = [0.0, 0.0]
        self.tau = self.args.tau_from
        for epoch in range(self.epoch, self.args.num_epochs):
            self._train_epoch(epoch)
            eval_model = self.args.eval_model and (
                    ((epoch + 1) % self.args.eval_interval == 0) or
                    (epoch + 1 == self.args.num_epochs))
            if eval_model and dist.get_rank() == 0:
                acc_with, acc_without = self._eval_epoch(epoch)
                if sum(acc_with) > sum(best_acc_with):
                    best_acc_with = acc_with
                    best_epoch_with = epoch
                if sum(acc_without) > sum(best_acc_without):
                    best_acc_without = acc_without
                    best_epoch_without = epoch
                result_with = \
                    'Best model (w/): {:.2f}% (Mid), {:.2f}% (Top), Epoch {}.'.format(
                        best_acc_with[0] * 100, best_acc_with[1] * 100,
                        best_epoch_with + 1
                    )
                self.print_log(result_with)
                result_without = \
                    'Best model (w/o): {:.2f}% (Mid), {:.2f}% (Top), Epoch {}.'.format(
                        best_acc_without[0] * 100, best_acc_without[1] * 100,
                        best_epoch_without + 1
                    )
                self.print_log(result_without)

    def _train_epoch(self, epoch):
        self.print_log(f'Train Model Epoch: {epoch + 1}')
        self.model.train()
        self.dataset['train'].sampler.set_epoch(epoch)

        overall_losses = []

        timer = 0.0
        self.record_time()
        for batch_id, data in enumerate(self.dataset['train']):
            # data
            x = data['points'].float().to(self.output_dev)
            label = data['labels'].long().to(self.output_dev)
            normal = data['normals'].float().to(self.output_dev)
            weights = torch.FloatTensor(self.dataset['train'].dataset.weights['top'])
            # forward
            out = self.model(
                x,
                normals=normal,
                sampling=self.args.sampling,
                tau=self.args.tau,
                hard=self.args.hard_sampling,
            )
            # compute losses
            pred_labels = out['top_score'].reshape(-1, self.level_classes[-1])
            target = label[:, -1, :].reshape(-1)

            if self.has_dummy_class[-1]:
                valid = target != self.level_classes[-1] - 1
            else:
                valid = torch.ones_like(target, dtype=torch.bool)
            loss = F.cross_entropy(
                pred_labels[valid], target[valid],
                weight=weights.to(self.output_dev)
            )

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # statistic
            dist.barrier()
            loss = self.distributed_reduce_mean(loss)
            overall_losses.append(loss.item())

            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, '
                    'lr1: {:.5f}, lr2: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['train']), loss.item(),
                        self.optimizer.param_groups[0]['lr'],
                        self.optimizer.param_groups[1]['lr']
                    ))
            timer += self.tick()
        self.scheduler.step()

        mean_loss = np.mean(overall_losses)
        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log('Time consumption: {:.1f} min'.format(timer / 60.0))

        if self.args.save_model and (epoch + 1) % self.args.save_interval == 0:
            model_path = os.path.join(self.model_path, f'model{epoch + 1}.pt')
            self.save_weights(
                epoch, self.model, self.optimizer, self.scheduler, model_path
            )
        if self.args.use_tensorboard and dist.get_rank() == 0:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/overall_loss', mean_loss, epoch)
        return mean_loss

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.print_log(f'Eval Model Epoch: {epoch + 1}')
        self.model.eval()

        loss_values = []
        pred_scores = {'top': [], 'mid': []}
        true_scores = {'top': [], 'mid': []}
        raw_scores = {'top': [], 'mid': []}
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
            # statistic
            # gather prediction and ground-truth
            for i, level in enumerate(['mid', 'top']):
                raw_score = out[f'{level}_score'].reshape(-1, self.level_classes[i])
                raw_scores[level].append(raw_score)
                true_label = label[:, i, :].reshape(-1)
                pred_label = raw_score.max(dim=1)[1]
                pred_scores[level].append(pred_label)
                true_scores[level].append(true_label)

            if self.has_dummy_class[-1]:
                valid = true_scores['top'][-1] != self.level_classes[-1] - 1
            else:
                valid = torch.ones_like(true_scores['top'][-1], dtype=torch.bool)
            loss = F.cross_entropy(
                raw_scores['top'][-1][valid], true_scores['top'][-1][valid]
            )
            # loss = self.distributed_reduce_mean(loss)
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['test']),
                        loss.item(), self.optimizer.param_groups[0]['lr']
                    ))
            timer += self.tick()

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
            if level == 'mid':
                fuck_pred = pred_scores[level]
                count = []
                for i in range(0, fuck_pred.shape[0], 2048):
                    cur_fuck_pred = fuck_pred[i:i+2048]
                    count.append(len(set(cur_fuck_pred)))
                bar = np.bincount(count)
                fig = plt.figure(figsize=(15, 15))
                plt.bar(range(len(bar)), bar)
                plt.savefig(os.path.join(self.args.save_dir, f'{epoch+1}.jpeg'))
                plt.close(fig)
            if self.has_dummy_class[0]:
                acc_without[level] = utils.clustering_acc(
                    true_scores[level], pred_scores[level],
                    self.args.num_points, self.level_classes[0] - 1
                )['accuracy']
            else:
                acc_without[level] = acc_with[level]

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

        if self.args.use_tensorboard and dist.get_rank() == 0:
            self.print_log('Writing to TensorBoard...')
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('test/supervised_loss', mean_loss, epoch)
                for level in ['mid', 'top']:
                    writer.add_scalar(
                        f'test/{level}_acc_with_dummy', acc_with[level], epoch
                    )
                    writer.add_scalar(
                        f'test/{level}_acc_without_dummy',
                        acc_without[level], epoch
                    )
                    writer.add_scalar(
                        f'test/{level}_mIoU_with_dummy', iou_with[level], epoch
                    )
        self.print_log('Time consumption: {:.1f} min'.format(timer / 60.0))
        return list(acc_with.values()), list(acc_without.values())
