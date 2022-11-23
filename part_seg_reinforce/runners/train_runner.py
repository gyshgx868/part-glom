import os
import shutil
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import part_seg_reinforce.tools.utils as utils

from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from torch.distributions.one_hot_categorical import OneHotCategorical

from part_seg_reinforce.datasets.part_net_seg import PartNetMultiLevelSegmentation
from part_seg_reinforce.models.net import MiddleModel
from part_seg_reinforce.models.net import TopModel
from part_seg_reinforce.runners.runner import Runner


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
            self.print_log(f"Test data loaded: {len(self.dataset['test'])} samples.")

    def load_model(self):
        self.model1 = MiddleModel(self.level_classes).to(self.output_dev)
        self.model2 = TopModel(self.level_classes).to(self.output_dev)

    def initialize_model(self):
        if self.args.weights is not None:
            self.load_weights(self.args.weights, self.args.ignore_weights)

    def run(self):
        # save initialized weights
        model_path = os.path.join(self.model_path, f'model_init.pt')
        self.save_weights(-1, model_path)

        best_epoch_with = -1
        best_epoch_without = -1
        best_acc_with = [0.0, 0.0]
        best_iou_with = [0.0, 0.0]
        best_acc_without = [0.0, 0.0]

        self.B = 1.0
        for epoch in range(self.epoch, self.args.num_epochs):
            self._train_epoch(epoch)
            eval_model = self.args.eval_model and (
                    ((epoch + 1) % self.args.eval_interval == 0) or
                    (epoch + 1 == self.args.num_epochs))
            if eval_model:
                acc_with, acc_without, iou_with = self._eval_epoch(epoch)
                if sum(acc_with) > sum(best_acc_with):
                    best_acc_with = acc_with
                    best_iou_with = iou_with
                    best_epoch_with = epoch
                    best_path = os.path.join(self.model_path, 'model_best_with.pt')
                    model_path = os.path.join(self.model_path, f'model{epoch+1}.pt')
                    shutil.copy(model_path, best_path)
                if sum(acc_without) > sum(best_acc_without):
                    best_acc_without = acc_without
                    best_epoch_without = epoch
                    best_path = os.path.join(self.model_path, 'model_best_without.pt')
                    model_path = os.path.join(self.model_path, f'model{epoch+1}.pt')
                    shutil.copy(model_path, best_path)
                result_with = \
                    'Best model (Acc. w/): {:.2f}% (Mid), {:.2f}% (Top), Epoch {}.'.format(
                        best_acc_with[0] * 100, best_acc_with[1] * 100,
                        best_epoch_with + 1
                    )
                self.print_log(result_with)
                result_with = \
                    'Best model (IoU. w/): {:.2f}% (Mid), {:.2f}% (Top), Epoch {}.'.format(
                        best_iou_with[0] * 100, best_iou_with[1] * 100,
                        best_epoch_with + 1
                    )
                self.print_log(result_with)
                result_without = \
                    'Best model (Acc. w/o): {:.2f}% (Mid), {:.2f}% (Top), Epoch {}.'.format(
                        best_acc_without[0] * 100, best_acc_without[1] * 100,
                        best_epoch_without + 1
                    )
                self.print_log(result_without)

    def _train_epoch(self, epoch):
        self.print_log(f'Train Model Epoch: {epoch + 1}')
        self.model1.train()
        self.model2.train()

        supervised_losses = []
        policy_grads = []
        timer = 0.0
        self.B = 1.0
        self.record_time()
        for batch_id, data in enumerate(self.dataset['train']):
            # data
            x = data['points'].float().to(self.output_dev)
            label = data['labels'].long().to(self.output_dev)
            normal = data['normals'].float().to(self.output_dev)
            weights = torch.FloatTensor(self.dataset['train'].dataset.weights['top'])

            # forward
            out1 = self.model1(x, normals=normal, sampling='categorical')
            out5 = out1['out5']
            embed = out1['embeddings']
            target = label[:, -1, :].reshape(-1)
            indices = target.unsqueeze(1)

            assert len(out1['logits'].size()) == 2
            sampler = OneHotCategorical(logits=out1['logits'])
            mid_probs = []
            mid_log_probs = []
            top_probs = []
            top_logits = []
            for s in range(self.args.num_samples):
                current_z = sampler.sample()
                mid_probs.append(current_z)
                mid_log_probs.append(sampler.log_prob(current_z))
                out2 = self.model2(
                    out5, embed, current_z.detach(),
                    normals=normal
                )
                probs = torch.gather(out2['probs'], dim=1, index=indices)
                probs = probs.squeeze(1).detach()
                top_probs.append(probs)
                top_logits.append(out2['logits'])
            pred = torch.mean(torch.stack(top_logits), dim=0)

            y = torch.stack(top_probs)  # (L, B*N)
            z_hat = torch.mean(y, dim=0)  # (B*N,)
            log_z = torch.stack(mid_log_probs)  # (L, B*N)
            policy_grad = -log_z * (y / z_hat - self.B)  # (L, B*N)
            policy_grad = torch.mean(policy_grad, dim=0)  # (B*N,)
            policy_grad = torch.mean(policy_grad)  # scalar
            # compute losses
            if self.has_dummy_class[-1]:
                valid = target != self.level_classes[-1] - 1
            else:
                valid = torch.ones_like(target, dtype=torch.bool)
            loss_s = F.cross_entropy(
                pred[valid], target[valid], weight=weights.to(self.output_dev)
            )
            # backward
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss_s.backward(retain_graph=True)
            policy_grad.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            # statistic
            supervised_losses.append(loss_s.item())
            policy_grads.append(policy_grad.item())

            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, '
                    'lr1: {:.5f}, lr2: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['train']),
                        loss_s.item(),
                        self.optimizer1.param_groups[0]['lr'],
                        self.optimizer2.param_groups[0]['lr'],
                    ))
            timer += self.tick()
        self.scheduler1.step()
        self.scheduler2.step()

        mean_loss = np.mean(supervised_losses)
        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log('Time consumption: {:.1f} min'.format(timer / 60.0))

        if self.args.save_model and (epoch + 1) % self.args.save_interval == 0:
            model_path = os.path.join(self.model_path, f'model{epoch + 1}.pt')
            self.save_weights(epoch, model_path)
        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/overall_loss', mean_loss, epoch)
                mean_s = np.mean(supervised_losses)
                writer.add_scalar('train/supervised_loss', mean_s, epoch)
                mean_p = np.mean(policy_grads)
                writer.add_scalar('train/policy_grad', mean_p, epoch)
        return mean_loss

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.print_log(f'Eval Model Epoch: {epoch + 1}')
        self.model1.eval()
        self.model2.eval()

        loss_values = []
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
            # loss = self.distributed_reduce_mean(loss)
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr1: {:.5f}, lr2: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['test']),
                        loss.item(),
                        self.optimizer1.param_groups[0]['lr'],
                        self.optimizer2.param_groups[0]['lr'],
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
            # ==================================================================
            matched_pred = pred_scores[level]
            count = []
            for i in range(0, matched_pred.shape[0], 2048):
                cur_pred = matched_pred[i:i+2048]
                count.append(len(set(cur_pred)))
            bar = np.bincount(count)
            fig = plt.figure(figsize=(15, 15))
            plt.bar(range(len(bar)), bar)
            plt.savefig(os.path.join(self.args.save_dir, f'{epoch+1}.jpeg'))
            plt.close(fig)
            # ==================================================================
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

        if self.args.use_tensorboard:
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
        return list(acc_with.values()), list(acc_without.values()), list(iou_with.values())
