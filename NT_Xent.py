import torch
import torch.nn as nn
import torch.nn.functional as F
#这个才是对比学习的loss，考虑了正负样本对2N，来源于论文的loss===================
def NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=True, temperature=0.5):
    LARGE_NUM = 1e9
    batch_size = hidden_features_transform_1.size(0)
    
    h1 = hidden_features_transform_1
    h2 = hidden_features_transform_2

    if normalize:
        h1 = F.normalize(h1, dim=1, p=2)
        h2 = F.normalize(h2, dim=1, p=2)

    labels = torch.arange(batch_size, device=hidden_features_transform_1.device)
    masks = torch.eye(batch_size, device=hidden_features_transform_1.device)

    logits_aa = torch.mm(h1, h1.t()) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.mm(h2, h2.t()) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.mm(h1, h2.t()) / temperature
    logits_ba = torch.mm(h2, h1.t()) / temperature

    combined_logits_a = torch.cat((logits_ab, logits_aa), dim=1)
    combined_logits_b = torch.cat((logits_ba, logits_bb), dim=1)

    loss = F.cross_entropy(combined_logits_a, labels) + F.cross_entropy(combined_logits_b, labels)

    return loss

#这个loss，考虑了N，1和1’，2’，3’...N'的关系===================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # 构造标签（每个样本与其自身匹配，其他样本都为负样本）
        labels = torch.arange(0, batch_size, device=z1.device)
        labels = torch.cat((labels[batch_size // 2:], labels[:batch_size // 2]))

        # 计算相似度矩阵
        similarities = torch.mm(z1, z2.T) / self.temperature

        # 计算损失
        loss = self.criterion(similarities, labels)

        return loss
#这个loss，考虑了N-1，1和2’，3’...N'的关系===================
class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # 计算余弦相似度
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        similarity = torch.matmul(z1, z2.T) / self.temperature

        # 构建对角线掩码，将对角线元素排除在计算中
        mask = torch.eye(similarity.size(0), device=similarity.device)

        # 计算NT-Xent损失
        loss = -torch.log(torch.exp(similarity) / torch.exp(similarity).sum(1, keepdim=True))
        loss = loss * (1 - mask)  # 掩码对角线元素
        loss = loss.sum() / (similarity.size(0)-1)  # 归一化损失

        return loss


z1,z2 = torch.randn(10, 128), torch.randn(10, 128)
loss_fn1 = NTXentLoss(temperature=0.5)
loss_fn2 = SimCLRLoss(temperature=0.5)
loss1 = loss_fn1(z1, z2)
loss2 = loss_fn2(z1, z2)
loss3 = NT_Xent_loss(z1, z2, temperature=0.5)
print(loss1, loss2,loss3)