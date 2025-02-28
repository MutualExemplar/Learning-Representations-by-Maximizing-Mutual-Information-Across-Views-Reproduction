import os
import logging
import sys
import torch

def update_pseudo_labels(model_F1, model_F2, model_F3, unlabeled_loader, threshold=0.5):
    model_F1.eval()
    model_F2.eval()
    model_F3.eval()

    pseudo_labels = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Batch length: {len(batch)}")
            print(f"Batch shape: {[b.shape for b in batch]}")

            if len(batch) == 4:
                images_U1, images_U2, images_U3, _ = batch  # Ignore extra label/mask
            else:
                images_U1, images_U2, images_U3 = batch

            # Stack images
            images_U1 = torch.stack(images_U1, dim=0).cuda()
            images_U2 = torch.stack(images_U2, dim=0).cuda()
            images_U3 = torch.stack(images_U3, dim=0).cuda()

            # Ensure all images have 3 channels
            if images_U1.shape[1] == 1:
                images_U1 = images_U1.expand(-1, 3, -1, -1)
            if images_U2.shape[1] == 1:
                images_U2 = images_U2.expand(-1, 3, -1, -1)
            if images_U3.shape[1] == 1:
                images_U3 = images_U3.expand(-1, 3, -1, -1)

            # Forward pass for pseudo-labeling
            pred_U1, _ = model_F1(images_U1)
            pred_U2, _ = model_F2(images_U2)
            pred_U3, _ = model_F3(images_U3)

            # Compute pseudo-labels as the average of model predictions
            pseudo_label = (torch.sigmoid(pred_U1) + torch.sigmoid(pred_U2) + torch.sigmoid(pred_U3)) / 3
            pseudo_labels.append(pseudo_label)

    # 🟢 Fix: Ensure pseudo_labels is correctly stacked
    if len(pseudo_labels) > 0:
        pseudo_labels = torch.cat(pseudo_labels, dim=0)  # Stack correctly
        print(f"✅ Generated pseudo_labels shape after concatenation: {pseudo_labels.shape}")
    else:
        print(f"⚠ Warning: No pseudo-labels were generated. Returning default tensor.")
        pseudo_labels = torch.zeros((1, 1, 512, 288), device='cuda')  # Safe tensor

    return pseudo_labels

def create_exp_dir(path, desc='Experiment directory: {}'):
    
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def create_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(log_dir, log_name='experiment.log'):
    
    create_exp_dir(log_dir)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file_path = os.path.join(log_dir, log_name)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=[
            logging.FileHandler(log_file_path),  # Save logs to file
            logging.StreamHandler(sys.stdout)    # Print logs in console
        ]
    )

    logger = logging.getLogger("MutualExemplar")
    return logger

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="latest.pth"):
    """
    Saves model checkpoint and tracks the best model.

    Args:
        state (dict): Model state dictionary.
        is_best (bool): If True, saves the best model separately.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): Name of the file to save.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_filepath)
        print(f"✅ Best model saved to {best_filepath}")

# **🔹 Load Checkpoint**
def load_checkpoint(model_F1, model_F2, model_F3, optimizer, checkpoint_dir="checkpoints", filename="best.pth"):
    """
    Loads the best saved checkpoint.

    Args:
        model_F1, model_F2, model_F3: Mutual Exemplar models.
        optimizer: Training optimizer.
        checkpoint_dir: Directory where checkpoints are stored.
        filename: Name of the checkpoint file.

    Returns:
        last_epoch (int): Last saved epoch.
        best_dice (float): Best recorded DSC.
    """
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model_F1.load_state_dict(checkpoint["model_F1"])
        model_F2.load_state_dict(checkpoint["model_F2"])
        model_F3.load_state_dict(checkpoint["model_F3"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["epoch"]
        best_dice = checkpoint["best_dice"]
        print(f"✅ Loaded checkpoint '{filename}' (Epoch {last_epoch}, Best DSC: {best_dice:.4f})")
        return last_epoch, best_dice
    else:
        print(f"⚠ No checkpoint found at '{filepath}', starting from scratch.")
        return 0, 0.0

# **🔹 Update Pseudo-Labels for Unlabeled Data**
def update_pseudo_labels(model_F1, model_F2, model_F3, unlabeled_loader, threshold=0.5):
    model_F1.eval()
    model_F2.eval()
    model_F3.eval()

    pseudo_labels = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Batch length: {len(batch)}")
            print(f"Batch shape: {[b.shape for b in batch]}")

            if len(batch) == 4:
                images_U1, images_U2, images_U3, _ = batch  # Ignore extra label/mask
            else:
                images_U1, images_U2, images_U3 = batch

            # Ensure all images have 3 channels
            images_U1 = [img.expand(3, -1, -1) if img.shape[0] == 1 else img for img in images_U1]
            images_U2 = [img.expand(3, -1, -1) if img.shape[0] == 1 else img for img in images_U2]
            images_U3 = [img.expand(3, -1, -1) if img.shape[0] == 1 else img for img in images_U3]

            # Stack images
            images_U1 = torch.stack(images_U1, dim=0).cuda()
            images_U2 = torch.stack(images_U2, dim=0).cuda()
            images_U3 = torch.stack(images_U3, dim=0).cuda()

            # Forward pass for pseudo-labeling
            pred_U1, _ = model_F1(images_U1)
            pred_U2, _ = model_F2(images_U2)
            pred_U3, _ = model_F3(images_U3)

            # Compute pseudo-labels as the average of model predictions
            pseudo_label = (torch.sigmoid(pred_U1) + torch.sigmoid(pred_U2) + torch.sigmoid(pred_U3)) / 3
            pseudo_labels.append(pseudo_label)

            # 🟢 Ensure pseudo_labels is not empty before using .shape
            if len(pseudo_labels) == 0:
                print(f"⚠ Warning: No pseudo-labels were generated. Returning default tensor.")
                pseudo_labels = torch.zeros((1, 1, 512, 288), device='cuda')  # Safe default
            else:
                pseudo_labels = torch.cat(pseudo_labels)


            print(f"Generated pseudo_labels shape after concatenation: {pseudo_labels.shape}")

            return pseudo_labels
        
            pseudo_labels = []

            with torch.no_grad():
                for batch in unlabeled_loader:
                    print(f"Batch type: {type(batch)}")
                    print(f"Batch length: {len(batch)}")
                    print(f"Batch shape: {[b.shape for b in batch]}")

                    if len(batch) == 4:
                        images_U1, images_U2, images_U3, _ = batch  # Ignore extra label/mask
                    else:
                        images_U1, images_U2, images_U3 = batch

                    images_U1 = images_U1.to("cuda")
                    images_U2 = images_U2.to("cuda")
                    images_U3 = images_U3.to("cuda")

                    pred_U1, _ = model_F1(images_U1)
                    pred_U2, _ = model_F2(images_U2)
                    pred_U3, _ = model_F3(images_U3)

                    pseudo_label = (torch.sigmoid(pred_U1) + torch.sigmoid(pred_U2) + torch.sigmoid(pred_U3)) / 3
                    pseudo_labels.append(pseudo_label)

            # 🟢 Fix: Ensure pseudo_labels is correctly stacked
            if len(pseudo_labels) > 0:
                pseudo_labels = torch.cat(pseudo_labels, dim=0)  # Stack correctly
                print(f"✅ Generated pseudo_labels shape after concatenation: {pseudo_labels.shape}")
            else:
                print(f"⚠ Warning: No pseudo-labels were generated. Returning default tensor.")
                pseudo_labels = torch.zeros((1, 1, 512, 288), device='cuda')  # Safe tensor

            return pseudo_labels


