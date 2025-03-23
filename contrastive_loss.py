import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q_F1, q_F2, q_F3, pseudo_Yf):
        """
        Compute Contrastive Loss between feature maps.
        """
        tau = self.temperature

        # ✅ Normalize Feature Maps
        q_F1 = F.normalize(q_F1, p=2, dim=1)
        q_F2 = F.normalize(q_F2, p=2, dim=1)
        q_F3 = F.normalize(q_F3, p=2, dim=1)
        pseudo_Yf = F.normalize(pseudo_Yf, p=2, dim=1)

        # ✅ Match Batch Sizes
        batch_size = min(q_F1.shape[0], pseudo_Yf.shape[0], q_F2.shape[0], q_F3.shape[0])
        q_F1, q_F2, q_F3, pseudo_Yf = q_F1[:batch_size], q_F2[:batch_size], q_F3[:batch_size], pseudo_Yf[:batch_size]

        # ✅ Compute Similarity Scores (Batch-wise)
        pos_F1 = torch.exp(torch.clamp(torch.cosine_similarity(q_F1, pseudo_Yf, dim=1) / tau, min=-50, max=50))
        pos_F2 = torch.exp(torch.clamp(torch.cosine_similarity(q_F2, pseudo_Yf, dim=1) / tau, min=-50, max=50))
        pos_F3 = torch.exp(torch.clamp(torch.cosine_similarity(q_F3, pseudo_Yf, dim=1) / tau, min=-50, max=50))

        neg_F1 = torch.exp(torch.clamp(torch.cosine_similarity(q_F1, q_F3, dim=1) / tau, min=-50, max=50)).sum()
        neg_F2 = torch.exp(torch.clamp(torch.cosine_similarity(q_F2, q_F1, dim=1) / tau, min=-50, max=50)).sum()
        neg_F3 = torch.exp(torch.clamp(torch.cosine_similarity(q_F3, q_F2, dim=1) / tau, min=-50, max=50)).sum()

        # ✅ Ensure Negatives Have the Same Shape as Positives
        neg_F1 = neg_F1.expand_as(pos_F1)
        neg_F2 = neg_F2.expand_as(pos_F2)
        neg_F3 = neg_F3.expand_as(pos_F3)

        # ✅ Compute Contrastive Loss Per Batch
        loss_F1 = -torch.log(torch.clamp(pos_F1 / torch.maximum(pos_F1 + neg_F1, torch.tensor(1e-6, device=pos_F1.device)), min=1e-6))
        loss_F2 = -torch.log(torch.clamp(pos_F2 / torch.maximum(pos_F2 + neg_F2, torch.tensor(1e-6, device=pos_F2.device)), min=1e-6))
        loss_F3 = -torch.log(torch.clamp(pos_F3 / torch.maximum(pos_F3 + neg_F3, torch.tensor(1e-6, device=pos_F3.device)), min=1e-6))

        return (loss_F1 + loss_F2 + loss_F3).mean()
