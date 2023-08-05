"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.3.3
    Description	:
            定义公共函数
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import paddle
from typing import List, Optional, Tuple, Union

class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(save_path, model, optimizer, epoch_n):
    paddle.save({"model_dict": model.state_dict(), "optimizer_dict": optimizer.state_dict(), "epoch_n": epoch_n},
                save_path)


def load_model(save_path, model, optimizer=None):
    model_data = paddle.load(save_path)
    model.load_state_dict(model_data["model_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(model_data["optimizer_dict"])
    epoch_n = model_data["epoch_n"]
    return epoch_n


def calc_psnr(img1, img2):
    return 10. * paddle.log10(1. / paddle.mean((img1 - img2) ** 2))


def calc_psnr2(img1, img2):
    return 20. * paddle.log10(1. / paddle.sqrt(paddle.mean((img1 - img2) ** 2)))


@paddle.no_grad()
def make_grid(
        tensor: Union[paddle.Tensor, List[paddle.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
) -> paddle.Tensor:
    """
        Make a grid of images.

        Args:
            tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
                or a list of images all of the same size.
            nrow(int, optional): Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding (int, optional): amount of padding. Default: ``2``.
            normalize (bool, optional): If True, shift the image to the range (0, 1),
                by the min and max values specified by ``value_range``. Default: ``False``.
            value_range (tuple, optional): tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
        Returns:
            grid (Tensor): the tensor containing grid of images.
        """

    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)
    if tensor.ndim == 3:  # Add batch dimension if not present
        tensor = tensor.unsqueeze(0)

    if normalize:
        if value_range is not None:
            min_val, max_val = value_range
        else:
            min_val, max_val = paddle.min(tensor), paddle.max(tensor)

        tensor = (tensor - min_val) / (max_val - min_val)

    B, C, H, W = tensor.shape

    # Calculate the number of columns and rows in the grid
    nrow = min(nrow, B)
    ncol = (B - 1) // nrow + 1

    # Calculate the size of the grid image
    grid_height = H * ncol + padding * (ncol - 1)
    grid_width = W * nrow + padding * (nrow - 1)

    # Create a blank grid image
    grid = paddle.zeros((C, grid_height, grid_width), dtype=tensor.dtype)

    # Fill the grid image with individual images from the input tensor
    for i in range(B):
        row_idx = i // nrow
        col_idx = i % nrow
        start_y = row_idx * (H + padding)
        start_x = col_idx * (W + padding)
        grid[:, start_y:start_y + H, start_x:start_x + W] = tensor[i]

    return grid
