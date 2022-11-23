import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from torch.distributions import OneHotCategorical

from part_seg_reinforce.models.layers import PointClassifier
from part_seg_reinforce.models.layers import StackedLatentLinear
from part_seg_reinforce.models.layers import TNet


class MiddleModel(nn.Module):
    def __init__(self, level_classes):
        super(MiddleModel, self).__init__()
        self.level_classes = level_classes
        num_classes = self.level_classes[0]

        self.tnet3d = TNet(in_channels=3)
        self.tnetkd = TNet(in_channels=128)
        self.conv1 = torch.nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(128, 128, kernel_size=1)
        self.conv4 = torch.nn.Conv1d(128, 256, kernel_size=1)
        self.conv5 = torch.nn.Conv1d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.classifier = PointClassifier(512, num_classes)
        self.label_smoothing = pyg_nn.GCNConv(num_classes, num_classes)

    def _sample(self, logits, **kwargs):
        B, N, L = logits.size()
        logits = logits.reshape(B * N, L)
        if self.training:
            if 'sampling' not in kwargs.keys():
                raise ValueError('Please appoint sampling method.')
            if kwargs['sampling'] == 'softmax':
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
            else:
                m = OneHotCategorical(logits=logits)
                probs = m.sample()
                log_probs = m.log_prob(probs)
        else:
            indices = torch.argmax(logits, dim=-1, keepdim=True)
            probs = torch.zeros_like(logits)
            probs.scatter_(1, indices, 1.0)
            log_probs = probs
        return {'probs': probs, 'log_probs': log_probs}

    @staticmethod
    def _build_edge_index(x, k):
        B, N, C = x.size()
        x = x.reshape(B * N, C)
        batch = torch.arange(0, B, dtype=torch.long).reshape(B, 1)
        batch = batch.repeat(1, N).view(-1).to(x.device)
        edge_index = pyg_nn.knn_graph(x, k=k, batch=batch)
        return edge_index

    def _smooth_logits(self, logits, edge_index, edge_weight):
        B, N, C = logits.size()
        logits = logits.reshape(B * N, C)
        # (B*N, C), (2, E)
        BN, C = logits.size()
        # assert edge_index.size(1) % BN == 0  # k
        row, col = edge_index
        label = torch.argmax(logits, dim=-1)
        # compute the most frequent class for the neighbors
        one_hot_label = F.one_hot(label, C)[row]
        res = torch.zeros_like(logits, dtype=torch.long)
        index = col.repeat(C, 1).transpose(1, 0)
        res.scatter_add_(0, index, one_hot_label)
        freq = torch.argmax(res, dim=-1)
        # erase edges
        label, freq = label[row], freq[col]
        valid = label == freq
        new_row, new_col = row[valid], col[valid]
        new_edge_index = torch.stack((new_row, new_col))
        # logits = self.label_smoothing(logits, new_edge_index)
        valid_edge_weight = edge_weight[valid]
        logits = self.label_smoothing(
            logits, new_edge_index, edge_weight=valid_edge_weight
        )
        logits = logits.reshape(B, N, C)
        return logits

    @staticmethod
    def _filter_edge_by_normal(edge_index, normals, threshold=0.95):
        r, c = edge_index
        normal = normals.reshape(-1, 3)
        ni, nj = normal[r, :], normal[c, :]
        cos = torch.abs(F.cosine_similarity(ni, nj))
        valid_deg = cos > threshold
        r, c = r[valid_deg], c[valid_deg]
        edge_index = torch.stack((r, c))
        return edge_index

    @staticmethod
    def _get_edge_weight_by_normal(edge_index, normals):
        r, c = edge_index
        normal = normals.reshape(-1, 3)
        ni, nj = normal[r, :], normal[c, :]
        cos = torch.abs(F.cosine_similarity(ni, nj))
        return cos

    def forward(self, x, **kwargs):
        B, N, C = x.size()

        normals = kwargs['normals']
        edge_index = self._build_edge_index(x, k=20)

        x = x.permute(0, 2, 1)  # (B, C, N)
        trans_3d = self.tnet3d(x)
        x = x.permute(0, 2, 1)  # (B, N, C)
        x = torch.bmm(x, trans_3d)
        x = x.permute(0, 2, 1)  # (B, C, N)

        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_kd = self.tnetkd(out3)
        x = out3.permute(0, 2, 1)  # (B, N, C)
        x = torch.bmm(x, trans_kd)
        x = x.permute(0, 2, 1)  # (B, C, N)
        out4 = F.relu(self.bn4(self.conv4(x)))
        out5 = self.bn5(self.conv5(out4))

        x = out5.permute(0, 2, 1)  # (B, N, C)
        embed_edge = self._build_edge_index(x, k=20)
        x = x.permute(0, 2, 1)  # (B, C, N)
        combined = torch.cat((edge_index, embed_edge), dim=-1)
        combined = pyg.utils.coalesce(combined)
        edge_weight = self._get_edge_weight_by_normal(combined, normals)
        logits = self.classifier(x)
        logits = self._smooth_logits(logits, combined, edge_weight=edge_weight)
        sampled = self._sample(logits, **kwargs)
        logits = logits.reshape(B * N, -1)

        embeddings = torch.cat([out1, out2, out3, out4, out5], dim=1)

        return {
            'embeddings': embeddings, 'out5': out5, 'logits': logits,
            'probs': sampled['probs'], 'log_probs': sampled['log_probs']
        }


class TopModel(nn.Module):
    def __init__(self, level_classes):
        super(TopModel, self).__init__()
        self.level_classes = level_classes
        num_classes = self.level_classes[-1]
        print(self.level_classes)

        self.latent = StackedLatentLinear(
            in_channels=512,
            out_channels=64,
            num_classes=self.level_classes[0]
        )
        self.convs1 = torch.nn.Conv1d(1152, 256, kernel_size=1)
        self.convs2 = torch.nn.Conv1d(256, 256, kernel_size=1)
        self.convs3 = torch.nn.Conv1d(256, 128, kernel_size=1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

        self.classifier = PointClassifier(128, num_classes)

    def _multiply(self, probability, stacked_features, batch_size):
        BN, L = probability.size()
        N = BN // batch_size
        embeddings = torch.bmm(probability.unsqueeze(1), stacked_features)
        embeddings = embeddings.squeeze(1)
        embeddings = embeddings.reshape(batch_size, N, -1)
        embeddings = embeddings.permute(0, 2, 1)  # (B, C, N)
        return embeddings

    @staticmethod
    def _build_edge_index(x, k):
        B, N, C = x.size()
        x = x.reshape(B * N, C)
        batch = torch.arange(0, B, dtype=torch.long).reshape(B, 1)
        batch = batch.repeat(1, N).view(-1).to(x.device)
        edge_index = pyg_nn.knn_graph(x, k=k, batch=batch)
        return edge_index

    @staticmethod
    def _filter_edge_by_normal(edge_index, normals, threshold=0.95):
        r, c = edge_index
        normal = normals.reshape(-1, 3)
        ni, nj = normal[r, :], normal[c, :]
        cos = torch.abs(F.cosine_similarity(ni, nj))
        valid_deg = cos > threshold
        r, c = r[valid_deg], c[valid_deg]
        edge_index = torch.stack((r, c))
        return edge_index

    @staticmethod
    def _get_edge_weight_by_normal(edge_index, normals):
        r, c = edge_index
        normal = normals.reshape(-1, 3)
        ni, nj = normal[r, :], normal[c, :]
        cos = torch.abs(F.cosine_similarity(ni, nj))
        return cos

    def forward(self, out5, embed, probs, **kwargs):
        B, C, N = out5.size()
        embeddings = self.latent(out5)
        embeddings = self._multiply(probs, embeddings, B)
        embeddings = torch.cat([embed, embeddings], dim=1)  # residual

        embeddings = F.relu(self.bns1(self.convs1(embeddings)))
        embeddings = F.relu(self.bns2(self.convs2(embeddings)))
        embeddings = F.relu(self.bns3(self.convs3(embeddings)))

        logits = self.classifier(embeddings)
        logits = logits.reshape(B * N, -1)
        probs_top = F.softmax(logits, dim=-1)
        log_probs_top = F.log_softmax(logits, dim=-1)

        return {
            'embeddings': embeddings,
            'logits': logits,
            'probs': probs_top,
            'log_probs': log_probs_top
        }


def main():
    import part_seg_reinforce.tools.utils as utils
    level_classes = (30, 6)
    model1 = MiddleModel(level_classes=level_classes)
    model2 = TopModel(level_classes=level_classes)
    print(model1)
    print(model2)
    t1 = utils.get_total_parameters(model1)['Total']
    t2 = utils.get_total_parameters(model2)['Total']
    print(t1 + t2)
    print('Latent:', utils.get_total_parameters(model2.latent))
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    B, C, N = 2, 3, 1024
    CV = 1
    for _ in range(10):
        x = torch.rand(B, N, C)
        n = torch.rand(B, N, C)
        labels = torch.argmax(torch.rand(B*N, 6), dim=-1)
        indices = labels.unsqueeze(1)
        out1 = model1(x, sampling='categorical', normals=n)
        print('Output 1:')
        for k, v in out1.items():
            print(' ', k, v.size())
        probs, log_probs, out5 = out1['probs'], out1['log_probs'], out1['out5']
        embed = out1['embeddings']

        assert len(out1['logits'].size()) == 2
        sampler = OneHotCategorical(logits=out1['logits'])
        mid_probs = []
        mid_log_probs = []
        top_probs = []
        top_logits = []
        for s in range(5):
            current_z = sampler.sample()
            mid_probs.append(current_z)
            mid_log_probs.append(sampler.log_prob(current_z))
            out2 = model2(
                out5.detach(), embed.detach(), current_z.detach(), normals=n
            )
            probs = torch.gather(out2['probs'], dim=1, index=indices)
            probs = probs.squeeze(1).detach()
            top_probs.append(probs)
            top_logits.append(out2['logits'])
        pred = torch.mean(torch.stack(top_logits), dim=0)

        y = torch.stack(top_probs)  # (S, B*N)
        z_hat = torch.mean(y, dim=0)  # (B*N,)
        log_z = torch.stack(mid_log_probs)  # (S, B*N)

        policy_grad = -log_z * (y / z_hat - CV)
        policy_grad = torch.mean(policy_grad, dim=0)  # (B*N,)
        policy_grad = torch.mean(policy_grad)  # scalar

        loss = F.cross_entropy(pred, labels)

        opt1.zero_grad()
        opt2.zero_grad()

        loss.backward(retain_graph=True)
        policy_grad.backward()
        opt1.step()
        opt2.step()
        print('Loss:', loss.item())
        print('Policy Loss:', policy_grad.item())


if __name__ == '__main__':
    main()
