"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2022.12.19
    Description	:
    Reference	:
        Learning a Deep Convolutional Network for Image Super-Resolution    2014
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import paddle.nn as nn


class SRCNN(nn.Layer):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.Conv1 = nn.Conv2D(3, 64, 9, 1, 4)
        self.Conv2 = nn.Conv2D(64, 32, 3, 1, 1)
        self.Conv3 = nn.Conv2D(32, 3, 5, 1, 2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.Relu(self.Conv1(x))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)
        return out
