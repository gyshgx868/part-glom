import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from part_seg_pathwise.models.layers import PointClassifier
from part_seg_pathwise.models.layers import StackedLatentLinear
from part_seg_pathwise.models.layers import TNet


class PartGLOM(nn.Module):
    def __init__(self, level_classes, **kwargs):
        """
        :param level_classes: number of classes for each level
        """
        super(PartGLOM, self).__init__()
        self.level_classes = level_classes

        # for the middle level
        self.tnet3d = TNet(in_channels=3)
        self.tnetkd = TNet(in_channels=128)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.mid_classifier = PointClassifier(512, self.level_classes[0])
        self.label_smoothing = pyg_nn.GCNConv(
            self.level_classes[0], self.level_classes[0]
        )

        # for the top level
        self.top_latent = StackedLatentLinear(
            in_channels=512,
            out_channels=64,
            num_classes=self.level_classes[0]
        )
        self.top_convs1 = nn.Conv1d(1152, 256, kernel_size=1)
        self.top_convs2 = nn.Conv1d(256, 256, kernel_size=1)
        self.top_convs3 = nn.Conv1d(256, 128, kernel_size=1)
        self.top_bns1 = nn.BatchNorm1d(256)
        self.top_bns2 = nn.BatchNorm1d(256)
        self.top_bns3 = nn.BatchNorm1d(128)
        self.top_classifier = PointClassifier(128, self.level_classes[-1])

    def sample(self, logit, **kwargs):
        if self.training:
            if 'sampling' not in kwargs.keys():
                raise ValueError('Please appoint sampling method.')
            if kwargs['sampling'] == 'gumbel':
                tau = 1.0 if 'tau' not in kwargs.keys() else kwargs['tau']
                hard = False if 'hard' not in kwargs.keys() else kwargs['hard']
                sampled = F.gumbel_softmax(logit, tau=tau, hard=hard, dim=-1)
            elif kwargs['sampling'] == 'softmax':
                sampled = F.softmax(logit, dim=-1)
            else:
                raise ValueError('Unknown sampling method.')
        else:
            indices = torch.argmax(logit, dim=-1, keepdim=True)
            sampled = torch.zeros_like(logit)
            sampled.scatter_(1, indices, 1.0)
        return sampled

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

    def forward(self, x, **kwargs):
        """
        :param x: (B, N, C)
        :param kwargs:
        :return:
        """
        normals = kwargs['normals']
        B, N, C = x.size()
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
        logits_mid = self.mid_classifier(x)
        logits_mid = self._smooth_logits(logits_mid, combined, edge_weight)
        logits_mid = logits_mid.reshape(B * N, -1)

        sampled = self.sample(logits_mid, **kwargs)

        embed = torch.cat([out1, out2, out3, out4, out5], dim=1)

        embeddings = self.top_latent(out5)
        embeddings = self._multiply(sampled, embeddings, B)
        embeddings = torch.cat([embed, embeddings], dim=1)  # residual

        embeddings = F.relu(self.top_bns1(self.top_convs1(embeddings)))
        embeddings = F.relu(self.top_bns2(self.top_convs2(embeddings)))
        embeddings = F.relu(self.top_bns3(self.top_convs3(embeddings)))

        logits_top = self.top_classifier(embeddings)
        logits_top = logits_top.reshape(B * N, -1)

        return {
            'mid_embeddings': embed,  # (B, C, N)
            'top_embeddings': embeddings,  # (B, C, N)
            'top_score': logits_top,  # (B*N, C2)
            'mid_score': logits_mid,  # (B*N, C1)
            'out5': out5
        }

    def sample_inference(self, x, out5, embed, sampled):
        B, N, C = x.size()
        embeddings = self.top_latent(out5)
        embeddings = self._multiply(sampled, embeddings, B)
        embeddings = torch.cat([embed, embeddings], dim=1)  # residual
        embeddings = F.relu(self.top_bns1(self.top_convs1(embeddings)))
        embeddings = F.relu(self.top_bns2(self.top_convs2(embeddings)))
        embeddings = F.relu(self.top_bns3(self.top_convs3(embeddings)))

        logits_top = self.top_classifier(embeddings)
        logits_top = logits_top.reshape(B * N, -1)

        return {
            'top_embeddings': embeddings,  # (B, C, N)
            'top_score': logits_top,  # (B*N, C2)
        }


def main():
    import time

    B, C, N = 2, 3, 1024
    x = torch.randn(B, N, C)
    normals = torch.rand_like(x)
    net = PartGLOM(level_classes=(30, 6))

    mid_parameters, top_parameters = [], []
    for k, v in net.named_parameters():
        if k.startswith('top_'):
            top_parameters.append(v)
        else:
            mid_parameters.append(v)
    optimizer = torch.optim.SGD(
        [
            {'params': top_parameters},
            {'params': mid_parameters, 'lr': 0.1}
        ],
        lr=0.01,
        weight_decay=1e-5,
        momentum=0.9
    )
    print(net)

    start = time.time()
    net.train()
    out = net(x, normals=normals, sampling='gumbel', tau=0.5)
    pred = out['top_score'].reshape(-1, 6)
    indices = torch.argmax(torch.rand_like(pred), dim=-1, keepdim=False)
    loss = F.cross_entropy(pred, indices)
    loss.backward()
    optimizer.step()
    print('param groups:', len(optimizer.param_groups))
    # for k, v in net.named_parameters():
    #     if v.grad is None:
    #         print(k, '=======')
    #     else:
    #         print(k, torch.norm(v.grad))
    print('Loss:', loss.item())
    end = time.time()
    print(f'{end - start}s.')
    for k, v in out.items():
        print(k, v.size() if isinstance(v, torch.FloatTensor) else type(v))


if __name__ == '__main__':
    main()
