import torch

def dice_coef(output, target, smooth=1e-6):
    """
    Compute Dice Coefficient (DSC) for binary and multi-class segmentation.
    - `output`: Model logits (not probabilities)
    - `target`: Ground truth mask
    """
    output = torch.sigmoid(output)  # Convert logits to probabilities
    
    # Handle multi-class segmentation
    if output.shape[1] > 1:  # If channels >1, it's multi-class
        output = torch.argmax(output, dim=1, keepdim=True)  # Convert to class labels
        target = torch.argmax(target, dim=1, keepdim=True)

    intersection = (output * target).sum(dim=(1, 2, 3))
    union = output.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()  # Mean across batch



def mean_absolute_error(output, target):
    
    return torch.nanmean(torch.abs(output - target)).item()



def evaluate_metrics(output, target):
    """
    Evaluate Dice Coefficient (DSC) and MAE.
    Supports both binary and multi-class segmentation.
    """
    return {
        "DSC": dice_coef(output, target),
        "MAE": mean_absolute_error(output, target)
    }
