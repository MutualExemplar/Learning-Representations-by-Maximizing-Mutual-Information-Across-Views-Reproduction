import torch
import torch.nn.functional as F
from contrastive_loss import ContrastiveLoss

### **Binary Cross-Entropy (BCE) Loss**
def bce_loss(pred, target):
    """
    Standard BCE loss with logits.
    """
    return F.binary_cross_entropy_with_logits(pred, target)

### **Intersection over Union (IoU) Loss**
def iou_loss(pred, target, epsilon=1e-6):
    """
    Computes standard IoU loss (1 - IoU).
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection

    return 1 - (intersection + epsilon) / (union + epsilon)

### **Combined BCE + IoU Loss**
def calc_loss(pred, target, bce_weight=0.5):
    """
    Compute BCE + IoU loss with adjustable weighting.
    """
    bce = bce_loss(pred, target)  # Standard BCE
    iou = iou_loss(torch.sigmoid(pred), target)  # ✅ Apply sigmoid here

    # Ensure total weight is 1
    iou_weight = 1 - bce_weight  
    return (bce * bce_weight) + (iou * iou_weight)

### **Supervised Loss for Labeled Data**
def loss_sup(logit_F1, logit_F2, logit_F3, labels):
    """
    Compute BCE + IoU loss for labeled data.
    """
    loss1 = calc_loss(logit_F1, labels)
    loss2 = calc_loss(logit_F2, labels)
    loss3 = calc_loss(logit_F3, labels)
    return (loss1 + loss2 + loss3) / 3  # Average across networks

### **Soft-Supervised Loss for Pseudo-Labels**
def loss_soft(logit_U1, logit_U2, logit_U3, pseudo_labels):
    """
    Compute BCE + IoU loss for unlabeled data (pseudo-labels).
    """
    loss1 = calc_loss(logit_U1, pseudo_labels)
    loss2 = calc_loss(logit_U2, pseudo_labels)
    loss3 = calc_loss(logit_U3, pseudo_labels)
    return loss1 + loss2 + loss3

### **Unsupervised Loss: Contrastive Loss + Soft Supervised Loss**
def loss_unsup(logit_U1, logit_U2, logit_U3, pseudo_labels, feat_F1, feat_F2, feat_F3, pseudo_Yf):
    """
    Compute Unsupervised Loss (Contrastive Loss + Soft Supervised Loss).
    """
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
    contrastive_loss = contrastive_loss_fn(feat_F1, feat_F2, feat_F3, pseudo_Yf)

    soft_supervised_loss = loss_soft(logit_U1, logit_U2, logit_U3, pseudo_labels)

    # ✅ Dynamically scale contrastive loss based on soft loss magnitude (avoiding instability)
    scaling_factor = soft_supervised_loss.mean().item() / max(contrastive_loss.mean().item(), 1e-6)
    contrastive_loss *= scaling_factor

    return soft_supervised_loss + contrastive_loss
