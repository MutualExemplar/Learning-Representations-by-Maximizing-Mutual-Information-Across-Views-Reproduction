import torch
import torch.nn.functional as F
from metrics import dice_coef
from contrastive_loss import ContrastiveLoss

BCE = torch.nn.BCELoss()

### **🔹 IoU + BCE Loss for Supervised & Soft Supervised Learning**
def iou_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)  # ✅ Convert logits to probabilities

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection

    return 1 - (intersection + epsilon) / (union + epsilon)


def weighted_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # ✅ Clamp logits to prevent log(0)
    pred = torch.clamp(pred, min=-10, max=10)  

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-6)  # ✅ Prevent division by zero

    inter = ((pred.sigmoid() * mask) * weight).sum(dim=(2, 3))  # ✅ Use sigmoid to convert logits to probabilities
    union = ((pred.sigmoid() + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def calc_loss(pred, target, bce_weight=0.5):
    """
    Compute BCE + IoU loss.
    """
    bce = weighted_loss(pred, target)
    iou = iou_loss(pred.sigmoid(), target)  # ✅ Use sigmoid activation

    # ✅ Ensure total weight is 1
    iou_weight = 1 - bce_weight  
    return (bce * bce_weight) + (iou * iou_weight)


### **🔹 Supervised Loss for Labeled Data**
def loss_sup(logit_F1, logit_F2, logit_F3, labels):
    """
    Compute Supervised Loss (BCE + IoU for each network).
    """
    loss1 = calc_loss(logit_F1, labels)
    loss2 = calc_loss(logit_F2, labels)
    loss3 = calc_loss(logit_F3, labels)
    return (loss1 + loss2 + loss3) / 3  # Sum across networks

### **🔹 Soft Supervised Loss for Pseudo-Labels (Unlabeled Data)**
def loss_soft(logit_U1, logit_U2, logit_U3, pseudo_labels):
    """
    Compute Soft Supervised Loss (BCE + IoU for unlabeled data using pseudo-labels).
    """
    loss1 = calc_loss(logit_U1, pseudo_labels)
    loss2 = calc_loss(logit_U2, pseudo_labels)
    loss3 = calc_loss(logit_U3, pseudo_labels)
    return loss1 + loss2 + loss3


def loss_unsup(logit_U1, logit_U2, logit_U3, pseudo_labels, feat_F1, feat_F2, feat_F3, pseudo_Yf):
    """
    Compute Unsupervised Loss (Contrastive Loss + Soft Supervised Loss).
    """
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
    contrastive_loss = contrastive_loss_fn(feat_F1, feat_F2, feat_F3, pseudo_Yf)

    soft_supervised_loss = loss_soft(logit_U1, logit_U2, logit_U3, pseudo_labels)

    # ✅ Dynamically scale contrastive loss based on soft loss magnitude
    scaling_factor = soft_supervised_loss.item() / (contrastive_loss.item() + 1e-6)
    contrastive_loss *= scaling_factor

    return soft_supervised_loss + contrastive_loss

