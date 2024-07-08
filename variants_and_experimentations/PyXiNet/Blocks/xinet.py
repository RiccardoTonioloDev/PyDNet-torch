"""
Code for XiNet (https://shorturl.at/mtHT0)

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
"""

import torch
import torch.nn as nn

from typing import Union, Tuple, Optional, List


def autopad(k: int, p: Optional[int] = None):
    """Implements padding to mimic "same" behaviour.
    Arguments
    ---------
    k : int
        Kernel size for the convolution.
    p : Optional[int]
        Padding value to be applied.

    """
    if (
        p is None
    ):  # if no padding was specified, it uses the kernel to determine the padding to apply
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class XiConv(nn.Module):
    """Implements XiNet's convolutional block as presented in the original paper.

    Arguments
    ---------
    c_in: int
        Number of input channels.
    c_out: int
        Number of output channels.
    kernel_size: Union[int, Tuple]
        Kernel size for the main convolution.
    stride: Union[int, Tuple]
        Stride for the main convolution.
    padding: Optional[Union[int, Tuple]]
        Padding that is applied in the main convolution.
    groups: Optional[int]
        Number of groups for the main convolution.
    act: Optional[bool]
        When True, uses SiLU activation function after
        the main convolution.
    gamma: Optional[float]
        Compression factor for the convolutional block.
    attention: Optional[bool]
        When True, uses attention.
    skip_tensor_in: Optional[bool]
        When True, defines broadcasting skip connection block.
    skip_res : Optional[List]
        Spatial resolution of the skip connection, such that
        average pooling is statically defined.
    skip_channels: Optional[int]
        Number of channels for the input block.
    pool: Optional[bool]
        When True, applies pooling after the main convolution.
    attention_k: Optional[int]
        Kernel for the attention module.
    attention_lite: Optional[bool]
        When True, uses efficient attention implementation.
    batchnorm: Optional[bool]
        When True, uses batch normalization inside the ConvBlock.
    dropout_rate: Optional[int]
        Dropout probability.
    skip_k: Optional[int]
        Kernel for the broadcast skip connection.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = True,
        gamma: Optional[float] = 4,
        attention: Optional[bool] = True,
        skip_tensor_in: Optional[bool] = True,
        skip_res: Optional[List] = None,
        skip_channels: Optional[int] = 1,
        pool: Optional[bool] = None,
        attention_k: Optional[int] = 3,
        attention_lite: Optional[bool] = True,
        batchnorm: Optional[bool] = True,
        dropout_rate: Optional[int] = 0,
        skip_k: Optional[int] = 1,
    ):
        super().__init__()
        self.compression = int(gamma)
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c_out // self.compression // 2
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate

        if skip_tensor_in:  # if we accept a skip broadcasting connection
            assert skip_res is not None, "Specifcy shape of skip tensor."
            self.adaptive_pooling = nn.AdaptiveAvgPool2d(  # we use the provided skip shape to perform an adaptive
                # avg pooling, to obtain the output sizes we want.
                (int(skip_res[0]), int(skip_res[1]))
            )
            # AdaptiveAvgPool2d allows us to define the output sizes of the tensor, and by itself
            # it decides the kernel size and the strides to use to obtain those output sizes.

        if (
            self.compression > 1
        ):  # if there is a compression factor, it will be applied using the pw-convolution
            self.compression_conv = nn.Conv2d(
                c_in,
                c_out // self.compression,
                1,
                1,
                groups=groups,
                bias=False,  # No bias term for the pw-convolution
            )
        self.main_conv = nn.Conv2d(
            c_out // self.compression if self.compression > 1 else c_in,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=autopad(kernel_size, padding),
            bias=False,  # No bias term even in the main convolution
        )
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )  # it uses the SiLU activation function if act is set to true.
        # However we can set our activation function if we want, otherwise the identity activation function
        # will be used.

        if attention:
            if (
                attention_lite
            ):  # if we choose the attention lite we can compress the attention convolution, similar
                # to what it was done in the first part of the XiConv block.
                self.att_pw_conv = nn.Conv2d(
                    c_out, self.attention_lite_ch_in, 1, 1, groups=groups, bias=False
                )
            self.att_conv = nn.Conv2d(
                c_out if not attention_lite else self.attention_lite_ch_in,
                c_out,
                attention_k,
                1,
                groups=groups,
                padding=autopad(attention_k, None),
                bias=False,
            )
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            # if it accepts a skip broadcast connection, it compresses the broadcast connection to the same
            # channel size of the output of the first pw-convolution, before using the main convolution.
            self.skip_conv = nn.Conv2d(
                skip_channels,
                c_out // self.compression,
                skip_k,
                1,
                groups=groups,
                padding=autopad(skip_k, None),
                bias=False,
            )
        if batchnorm:
            self.bn = nn.BatchNorm2d(c_out)
        if dropout_rate > 0:
            self.do = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """Computes the forward step of the XiNet's convolutional block.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        ConvBlock output. : torch.Tensor
        """
        s = None
        # skip connection
        if isinstance(
            x, list
        ):  # if we are passing a list, it means that the first element of the list is the real
            # input tensor, while the second one is the broadcasted tensor.

            # -----------------------------------------------------------
            # s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            # -----------------------------------------------------------

            # so here we resize in the dimensions of the broadcasted tensor
            s = self.adaptive_pooling(x[1])
            # and here we compress in the channels of the broadcasted tensor
            s = self.skip_conv(s)
            # finally we save as x the real input tensor
            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)  # compression of the channels if gamma > 1

        if s is not None:
            x = (
                x + s
            )  # if the skipped broadcast tensor was passed, we apply the skip connection

        if self.pool:  # if a pooling operation was set, then it applies it
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)  # applying the main convolution
        if self.batchnorm:  # if a normal batch operation was set, then it applies it
            x = self.bn(x)
        x = self.act(x)  # it applies the activation function after the main convolution

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in = self.att_pw_conv(x)
            else:
                att_in = x
            y = self.att_act(self.att_conv(att_in))
            x = x * y  # applying attention with an element-wise multiplication

        if self.dropout_rate > 0:
            x = self.do(x)  # doing a dropout iteration if set

        return x


class XiNet(nn.Module):
    """Defines a XiNet.

    Arguments
    ---------
    input_shape : List
        Shape of the input tensor.
    alpha: float
        Width multiplier.
    gamma : float
        Compression factor.
    num_layers : int = 5
        Number of convolutional blocks.
    num_classes : int
        Number of classes. It is used only when include_top is True.
    include_top : Optional[bool]
        When True, defines an MLP for classification.
    base_filters : int
        Number of base filters for the ConvBlock.
    return_layers : Optional[List]
        Ids of the layers to be returned after processing the foward
        step.

    Example
    -------
    .. doctest::

        >>> from micromind.networks import XiNet
        >>> model = XiNet((3, 224, 224))
    """

    def __init__(
        self,
        input_shape: List,
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        num_classes=1000,
        include_top=False,
        base_filters: int = 16,
        return_layers: Optional[List] = None,
    ):
        super().__init__()

        self._layers = nn.ModuleList([])
        self.input_shape = torch.Tensor(input_shape)
        self.include_top = include_top
        self.return_layers = return_layers
        count_downsample = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_shape[0],
                int(base_filters * alpha),
                7,
                padding=7 // 2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(int(base_filters * alpha)),
            nn.SiLU(),
        )
        count_downsample += 1

        num_filters = [int(2 ** (base_filters**0.5 + i)) for i in range(0, num_layers)]
        skip_channels_num = int(base_filters * 2 * alpha)

        for i in range(
            len(num_filters) - 2
        ):  # Account for the last two layers separately
            self._layers.append(
                XiConv(
                    int(num_filters[i] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    pool=2,
                    skip_tensor_in=(i != 0),
                    skip_res=self.input_shape[1:] / (2**count_downsample),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )
            count_downsample += 1
            self._layers.append(
                XiConv(
                    int(num_filters[i + 1] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    skip_tensor_in=True,
                    skip_res=self.input_shape[1:] / (2**count_downsample),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )

        # Adding the last two layers with attention=False
        self._layers.append(
            XiConv(
                int(num_filters[-2] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )
        # count_downsample += 1
        self._layers.append(
            XiConv(
                int(num_filters[-1] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )

        if self.return_layers is not None:
            print(f"XiNet configured to return layers {self.return_layers}:")
            for i in self.return_layers:
                print(f"Layer {i} - {self._layers[i].__class__}")

        self.input_shape = input_shape
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(num_filters[-1] * alpha), num_classes),
            )

    def forward(self, x):
        """Computes the forward step of the XiNet.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Output of the network, as defined from
            the init. : Union[torch.Tensor, Tuple]
        """
        x = self.conv1(x)
        skip = None
        ret = []
        for layer_id, layer in enumerate(self._layers):
            if layer_id == 0:
                x = layer(x)
                skip = x
            else:
                x = layer([x, skip])

            if self.return_layers is not None:
                if layer_id in self.return_layers:
                    ret.append(x)

        if self.include_top:
            x = self.classifier(x)

        if self.return_layers is not None:
            return x, ret

        return x
