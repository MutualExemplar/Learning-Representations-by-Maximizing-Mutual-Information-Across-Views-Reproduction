import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        #self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
       #self.mask_dtype = torch.bool

     ### **🔹 Contrastive Loss for Feature Consistency**
    def forward(self, q_F1, q_F2, q_F3, pseudo_Yf):
        """
        Compute Contrastive Loss between feature maps.
        """
        tau = self.temperature
        
        q_F1 = torch.nn.functional.normalize(q_F1, p=2, dim=1)
        q_F2 = torch.nn.functional.normalize(q_F2, p=2, dim=1)
        q_F3 = torch.nn.functional.normalize(q_F3, p=2, dim=1)
        pseudo_Yf = torch.nn.functional.normalize(pseudo_Yf, p=2, dim=1)

        
        pos_F1 = torch.exp(torch.clamp(torch.cosine_similarity(q_F1, pseudo_Yf) / tau, min=-50, max=50))
        pos_F2 = torch.exp(torch.clamp(torch.cosine_similarity(q_F2, pseudo_Yf) / tau, min=-50, max=50))
        pos_F3 = torch.exp(torch.clamp(torch.cosine_similarity(q_F3, pseudo_Yf) / tau, min=-50, max=50))

        neg_F1 = torch.exp(torch.clamp(torch.cosine_similarity(q_F1, q_F3) / tau, min=-50, max=50)).sum()
        neg_F2 = torch.exp(torch.clamp(torch.cosine_similarity(q_F2, q_F1) / tau, min=-50, max=50).sum())
        neg_F3 = torch.exp(torch.clamp(torch.cosine_similarity(q_F3, q_F2) / tau, min=-50, max=50).sum())

        loss_F1 = -torch.log(torch.clamp(pos_F1 / (pos_F1 + neg_F1 + 1e-6), min=1e-6))
        loss_F2 = -torch.log(torch.clamp(pos_F2 / (pos_F2 + neg_F2 + 1e-6), min=1e-6))
        loss_F3 = -torch.log(torch.clamp(pos_F3 / (pos_F3 + neg_F3 + 1e-6), min=1e-6))

        return (loss_F1 + loss_F2 + loss_F3).mean()

#     def forward(self, feat_F1, feat_F2, feat_F3, pseudo_Yf):
        
#         target_size = (8, 8)  
#         batch_size = feat_F1.shape[0]

#         # print(f"feat_F1 shape before pooling: {feat_F1.shape}")
#         # print(f"feat_F2 shape before pooling: {feat_F2.shape}")
#         # print(f"feat_F3 shape before pooling: {feat_F3.shape}")
#         # print(f"pseudo_Yf shape before pooling: {pseudo_Yf.shape}")

#         # **Ensure pseudo_Yf batch size matches feat_F1, feat_F2, feat_F3**
#         if pseudo_Yf.shape[0] != batch_size:
#             print(f"⚠ Adjusting pseudo_Yf batch size from {pseudo_Yf.shape[0]} to {batch_size}")
#             repeat_factor = batch_size // pseudo_Yf.shape[0] + (batch_size % pseudo_Yf.shape[0] > 0)
#             pseudo_Yf = pseudo_Yf.repeat(repeat_factor, 1)[:batch_size]  # Dynamically match batch size


#         # **Reshape feature maps to 4D (batch, channels, height, width)**
#         if len(feat_F1.shape) == 2:  
#             feat_F1 = feat_F1.view(batch_size, -1, 1, 1)
#             feat_F2 = feat_F2.view(batch_size, -1, 1, 1)
#             feat_F3 = feat_F3.view(batch_size, -1, 1, 1)

#         # print(f"feat_F1 shape after reshaping: {feat_F1.shape}")
#         # print(f"feat_F2 shape after reshaping: {feat_F2.shape}")
#         # print(f"feat_F3 shape after reshaping: {feat_F3.shape}")

#         # **Apply Adaptive Pooling**
#         feat_F1 = F.adaptive_avg_pool2d(feat_F1, target_size)
#         feat_F2 = F.adaptive_avg_pool2d(feat_F2, target_size)
#         feat_F3 = F.adaptive_avg_pool2d(feat_F3, target_size)

#         # **Flatten feature maps for similarity computation**
#         feat_F1 = feat_F1.view(batch_size, -1)  
#         feat_F2 = feat_F2.view(batch_size, -1)
#         feat_F3 = feat_F3.view(batch_size, -1)        

#         # **Ensure pseudo_Yf has the correct feature dimension**
#         pseudo_Yf = pseudo_Yf.view(batch_size, -1)  

#         # 🔹 Fix: Zero-pad to match max feature dimension
#         max_dim = max(feat_F1.shape[1], pseudo_Yf.shape[1])
#         pseudo_Yf = F.pad(pseudo_Yf, (0, max_dim - pseudo_Yf.shape[1]))
#         feat_F1 = F.pad(feat_F1, (0, max_dim - feat_F1.shape[1]))
#         feat_F2 = F.pad(feat_F2, (0, max_dim - feat_F2.shape[1]))
#         feat_F3 = F.pad(feat_F3, (0, max_dim - feat_F3.shape[1]))

#         # print(f"feat_F1 shape final: {feat_F1.shape}")
#         # print(f"feat_F2 shape final: {feat_F2.shape}")
#         # print(f"feat_F3 shape final: {feat_F3.shape}")
#         # print(f"pseudo_Yf shape final: {pseudo_Yf.shape}")

#         # **Normalize features before cosine similarity**
#         feat_F1 = F.normalize(feat_F1, p=2, dim=1)
#         feat_F2 = F.normalize(feat_F2, p=2, dim=1)
#         feat_F3 = F.normalize(feat_F3, p=2, dim=1)
#         pseudo_Yf = F.normalize(pseudo_Yf, p=2, dim=1)

#         # **Compute Cosine Similarity**
#         pos_F1 = torch.exp(torch.cosine_similarity(feat_F1, pseudo_Yf, dim=1) / self.temperature)
#         pos_F2 = torch.exp(torch.cosine_similarity(feat_F2, pseudo_Yf, dim=1) / self.temperature)
#         pos_F3 = torch.exp(torch.cosine_similarity(feat_F3, pseudo_Yf, dim=1) / self.temperature)

#         # **Avoid log(0) issue**
#         contrastive_loss = -torch.log(torch.clamp(pos_F1 + pos_F2 + pos_F3, min=1e-6))

#         return contrastive_loss.mean()  # Ensure a single scalar is returned
