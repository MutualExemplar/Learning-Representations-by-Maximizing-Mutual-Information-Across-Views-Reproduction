import os
import torch
import torch.optim as optim
import argparse
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from model.mutual_exemplar_unet import MutualExemplarUNet
from loss import loss_sup, loss_soft, loss_unsup
from contrastive_loss import ContrastiveLoss
from data import image_loader
from utils import get_logger, save_checkpoint
from metrics import evaluate_metrics

# **🔹 Training Hyperparameters**
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
parser.add_argument('--trainsize', type=int, nargs=2, default=[512, 288], help='Training image size (H, W)')
parser.add_argument('--dataset', type=str, default='kvasir', help='Dataset name')
parser.add_argument('--ratio', type=float, default=0.1, help='Ratio of labeled data')
opt = parser.parse_args()

# **🔹 Device Configuration**
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs/kvasir/train/"
CHECKPOINT_DIR = os.path.join(LOG_DIR, "Checkpoints_5%")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # **🔹 Initialize Logger**
logger = get_logger(LOG_DIR)

# **🔹 Load Data**
train_loader_labeled, train_loader_unlabeled, val_loader = image_loader(
    image_root=f"data/{opt.dataset}/train/image/",
    gt_root=f"data/{opt.dataset}/train/mask/",
    val_img_root=f"data/{opt.dataset}/test/image/",
    val_gt_root=f"data/{opt.dataset}/test/mask/",
    batch_size=opt.batchsize,
    image_size=opt.trainsize,
    labeled_ratio=opt.ratio,
)
#print(f"Labeled DataLoader batch size: {train_loader_labeled.batch_size}")
#print(f"Unlabeled DataLoader batch size: {train_loader_unlabeled.batch_size}")

# **🔹 Initialize Models**
model_F1 = MutualExemplarUNet().to(DEVICE)
model_F2 = MutualExemplarUNet().to(DEVICE)
model_F3 = MutualExemplarUNet().to(DEVICE)

# **🔹 Loss & Optimizer**
contrastive_loss_fn = ContrastiveLoss().to(DEVICE)
optimizer = optim.Adam(
    list(model_F1.parameters()) + list(model_F2.parameters()) + list(model_F3.parameters()),
    lr=opt.lr,
    betas=(0.9, 0.999)
)

# Initialize pseudo-label storage and best loss tracking
pseudo_Yf_memory = None
pseudo_Yp_memory = None
best_supervised_loss = float("inf")  # Track the best loss

# **🔹 Training Loop**
best_dice = 0.0

for epoch in range(opt.epoch):
    model_F1.train()
    model_F2.train()
    model_F3.train()

    total_loss, sup_loss, soft_loss, cont_loss = 0, 0, 0, 0

    for i, (batch_labeled, batch_unlabeled) in enumerate(zip(train_loader_labeled, train_loader_unlabeled)):
        # **🔹 Process Labeled Data**
        images_F1, images_F2, images_F3, masks = batch_labeled
        images_F1, images_F2, images_F3, masks = (
            images_F1.to(DEVICE),
            images_F2.to(DEVICE),
            images_F3.to(DEVICE),
            masks.to(DEVICE),
        )

        # **🔹 Process Unlabeled Data**
        unlabeled_F1, unlabeled_F2, unlabeled_F3, *_ = batch_unlabeled
        unlabeled_F1, unlabeled_F2, unlabeled_F3 = (
            unlabeled_F1.to(DEVICE),
            unlabeled_F2.to(DEVICE),
            unlabeled_F3.to(DEVICE),
        )

        # **🔹 Forward Pass for Labeled Data**
        pred_F1, feat_F1 = model_F1(images_F1)
        pred_F2, feat_F2 = model_F2(images_F2)
        pred_F3, feat_F3 = model_F3(images_F3)

        # **🔹 Compute Supervised Loss**
        supervised_loss = loss_sup(pred_F1, pred_F2, pred_F3, masks)

        # **🔹 Forward Pass for Unlabeled Data**
        pred_U1, feat_U1 = model_F1(unlabeled_F1)
        pred_U2, feat_U2 = model_F2(unlabeled_F2)
        pred_U3, feat_U3 = model_F3(unlabeled_F3)

        # Compute pseudo-labels with confidence thresholding
        threshold = 0.7  # Confidence threshold

        # Feature map pseudo-label (projector output)
        pseudo_Yf = (feat_U1 + feat_U2 + feat_U3) / 3  # Average feature maps
        pseudo_Yf = pseudo_Yf / (pseudo_Yf.norm(p=2, dim=1, keepdim=True) + 1e-6)  # L2-normalization
        if pseudo_Yf_memory is not None:
            pseudo_Yf = pseudo_Yf_memory

        # Prediction pseudo-label (classifier output)
        pseudo_Yp = (torch.sigmoid(pred_U1) + torch.sigmoid(pred_U2) + torch.sigmoid(pred_U3)) / 3
        pseudo_Yp_hard = (pseudo_Yp > threshold).float()  # Hard labels where confident
        pseudo_Yp = pseudo_Yp_hard * pseudo_Yp + (1 - pseudo_Yp_hard) * pseudo_Yp  # Hybrid pseudo-labels
        if pseudo_Yp_memory is not None:
            pseudo_Yp = pseudo_Yp_memory

             
        def extract_features(feature_map):
            """
            Converts 4D feature maps into 2D feature vectors.
            Ensures the input tensor has the correct shape before applying pooling.
            """
            # 🔹 Print shape for debugging
            #print(f"Feature Map Shape Before Pooling: {feature_map.shape}")

            # 🔹 Ensure feature_map is 4D (B, C, H, W)
            if feature_map.dim() == 2:  
                feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)  # Convert (B, C) → (B, C, 1, 1)
            elif feature_map.dim() == 3:
                feature_map = feature_map.unsqueeze(0)  # Convert (C, H, W) → (1, C, H, W)
            elif feature_map.dim() < 2:
                raise ValueError(f"Unexpected feature map shape: {feature_map.shape}")

            # 🔹 Now apply Adaptive Average Pooling
            feature_map = F.adaptive_avg_pool2d(feature_map, (1, 1))  # Convert (B, C, H, W) → (B, C, 1, 1)
            feature_map = feature_map.squeeze(-1).squeeze(-1)  # Convert (B, C, 1, 1) → (B, C)

            # 🔹 Print shape after pooling
            #print(f"Feature Map Shape After Pooling: {feature_map.shape}")

            return feature_map
        
        pseudo_Yf = extract_features(pseudo_Yf)
        # ✅ Now compute contrastive loss with 2D feature vectors
        contrastive_loss = contrastive_loss_fn(feat_F1, feat_F2, feat_F3, pseudo_Yf)

        # **🔹 Compute Soft Supervised Loss for Unlabeled Data**
        soft_supervised_loss = loss_soft(pred_U1, pred_U2, pred_U3, pseudo_Yp)

        unsupervised_loss = loss_unsup(pred_U1, pred_U2, pred_U3, pseudo_Yp, feat_F1, feat_F2, feat_F3, pseudo_Yf)
        
        # **🔹 Compute Total Loss**
        loss = supervised_loss.mean() + soft_supervised_loss.mean() + contrastive_loss + unsupervised_loss.mean()

        # Compute supervised loss for current pseudo-labels
        current_supervised_loss = loss_sup(pred_F1, pred_F2, pred_F3, masks)

        # If new loss is lower, update pseudo-labels
        if current_supervised_loss.mean().item() < best_supervised_loss:
            best_supervised_loss = current_supervised_loss.mean().item()  
            pseudo_Yf_memory = pseudo_Yf.clone().detach()
            pseudo_Yp_memory = pseudo_Yp.clone().detach()
           
        optimizer.zero_grad()
        supervised_loss.mean().backward()
        
        torch.nn.utils.clip_grad_norm_(model_F1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model_F2.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model_F3.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (i + 1) % 5 == 0:  
            torch.cuda.empty_cache()

        total_loss += loss.mean().item()
        sup_loss += supervised_loss.mean().item()
        soft_loss += soft_supervised_loss.mean().item()
        cont_loss += contrastive_loss.mean().item()


    # **🔹 Validation**
    with torch.no_grad():
        model_F1.eval()
        model_F2.eval()
        model_F3.eval()

        dice_scores = []

        for val_batch in val_loader:
            images, masks = val_batch
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            pred_F1, _ = model_F1(images)
            pred_F2, _ = model_F2(images)
            pred_F3, _ = model_F3(images)

            final_pred = (pred_F1 + pred_F2 + pred_F3) / 3  # Average predictions

            metrics = evaluate_metrics(final_pred, masks)
            dice_scores.append(metrics["DSC"])

    avg_dice = sum(dice_scores) / len(dice_scores)
    logger.info(f"Epoch [{epoch + 1}/{opt.epoch}], Loss: {total_loss:.4f}, DSC: {avg_dice:.4f}")

     
    torch.save(model_F1.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_1.pth'))
    torch.save(model_F2.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_2.pth'))
    torch.save(model_F3.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_3.pth'))
    
    # **🔹 Save Best Model**
    is_best = avg_dice > best_dice
    if is_best:
        best_dice = avg_dice
        
        torch.save(model_F1.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_1.pth'))
        torch.save(model_F2.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_2.pth'))
        torch.save(model_F3.state_dict(), os.path.join(CHECKPOINT_DIR, 'Model_3.pth'))

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "model_F1": model_F1.state_dict(),
            "model_F2": model_F2.state_dict(),
            "model_F3": model_F3.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dice": best_dice,
        },
        is_best,
        CHECKPOINT_DIR,
    )

logger.info("Training complete!")
