import torch

def feature_transform_reguliarzer(trans):
    """
    Regularizes the feature transformations by enforcing them to be close to orthogonal matrices.
    This helps preserve geometric structure in data, such as in point cloud processing.

    Args:
        trans (torch.Tensor): The transformation matrices to be regularized.

    Returns:
        torch.Tensor: The regularization loss, encouraging the transformation to be orthogonal.
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
