# import torch
# import torch.nn as nn
#
# class ConvTemporalGraphical(nn.Module):
#     # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
#     r"""The basic module for applying a graph convolution.
#     Args:
#         in_channels (int): Number of channels in the input sequence data
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int): Size of the graph convolving kernel
#         t_kernel_size (int): Size of the temporal convolving kernel
#         t_stride (int, optional): Stride of the temporal convolution. Default: 1
#         t_padding (int, optional): Temporal zero-padding added to both sides of
#             the input. Default: 0
#         t_dilation (int, optional): Spacing between temporal kernel elements.
#             Default: 1
#         bias (bool, optional): If ``True``, adds a learnable bias to the output.
#             Default: ``True``
#     Shape:
#         - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
#         - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
#         - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
#         - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
#         super(ConvTemporalGraphical, self).__init__()
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(
#             in_channels, out_channels,
#             kernel_size=(t_kernel_size, 1),
#             padding=(t_padding, 0),
#             stride=(t_stride, 1),
#             dilation=(t_dilation, 1),
#             bias=bias)
#
#     def forward(self, x, A):
#         # Log shape BEFORE conv
#         print(f"[ConvTemporalGraphical] input x: {x.shape}  | expected in_channels={self.conv.in_channels}")
#         assert A.size(0) == self.kernel_size
#
#         x = self.conv(x)
#         print(f"[ConvTemporalGraphical] after conv x: {x.shape}")
#
#         x = torch.einsum('nctv,tvw->nctw', (x, A))  # spatial graph op
#         print(f"[ConvTemporalGraphical] after einsum x: {x.shape}")
#         return x.contiguous(), A
#
#
# class st_gcn(nn.Module):
#     r"""Applies a spatial temporal graph convolution over an input graph sequence.
#     Args:
#         in_channels (int): Number of channels in the input sequence data
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
#         stride (int, optional): Stride of the temporal convolution. Default: 1
#         dropout (int, optional): Dropout rate of the final output. Default: 0
#         residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
#     Shape:
#         - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
#         - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
#         - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
#         - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False,
#                  stride=1, dropout=0, residual=True):
#         super(st_gcn, self).__init__()
#         assert len(kernel_size) == 2
#         assert kernel_size[0] % 2 == 1
#         padding = ((kernel_size[0] - 1) // 2, 0)
#         self.use_mdn = use_mdn
#
#         self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
#
#         self.tcn = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#             nn.PReLU(),
#             nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
#                       (stride, 1), padding),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout(dropout, inplace=True),
#         )
#
#         if not residual:
#             self.residual = lambda x: 0
#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x
#         else:
#             self.residual = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
#                 nn.BatchNorm2d(out_channels),
#             )
#
#         self.prelu = nn.PReLU()
#
#     def forward(self, x, A):
#         print(f"[st_gcn] input x: {x.shape}")
#         res = self.residual(x)
#         print(f"[st_gcn] residual: {res.shape if isinstance(res, torch.Tensor) else 'None'}")
#
#         x, A = self.gcn(x, A)
#         print(f"[st_gcn] after gcn x: {x.shape}")
#
#         x = self.tcn(x) + res
#         print(f"[st_gcn] after tcn x: {x.shape}")
#
#         if not self.use_mdn:
#             x = self.prelu(x)
#
#         return x, A
#
#
# class social_stgcnn(nn.Module):
#     def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5,
#                  seq_len=8, pred_seq_len=12, kernel_size=3):
#         super(social_stgcnn, self).__init__()
#         self.n_stgcnn = n_stgcnn
#         self.n_txpcnn = n_txpcnn
#
#         self.st_gcns = nn.ModuleList()
#         self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
#         for j in range(1, n_stgcnn):
#             self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))
#
#         self.tpcnns = nn.ModuleList()
#         self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
#         for j in range(1, n_txpcnn):
#             self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
#         self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)
#
#         self.prelus = nn.ModuleList()
#         for j in range(n_txpcnn):
#             self.prelus.append(nn.PReLU())
#
#     def forward(self, v, a):
#         print(f"[social_stgcnn] initial v: {v.shape}")
#         for k in range(self.n_stgcnn):
#             print(f"[social_stgcnn] --- st_gcn block {k} ---")
#             v, a = self.st_gcns[k](v, a)
#
#         print(f"[social_stgcnn] before tpcnn reshape v: {v.shape}")
#         v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # (N, T, C, V)
#         print(f"[social_stgcnn] reshaped for tpcnn: {v.shape}")
#
#         v = self.prelus[0](self.tpcnns[0](v))
#         print(f"[social_stgcnn] after tpcnn[0]: {v.shape}")
#
#         for k in range(1, self.n_txpcnn - 1):
#             v = self.prelus[k](self.tpcnns[k](v)) + v
#             print(f"[social_stgcnn] after tpcnn[{k}]: {v.shape}")
#
#         v = self.tpcnn_ouput(v)
#         print(f"[social_stgcnn] after tpcnn_output: {v.shape}")
#
#         v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # (N, C, T, V)
#         print(f"[social_stgcnn] output v shape: {v.shape}")
#
#         return v, a
import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = spatial_kernel_size

        # print(f"1 [ConvTemporalGraphical.__init__] self.kernel_size set to: {kernel_size}")
        # super(ConvTemporalGraphical, self).__init__()
        # print(f"2 [ConvTemporalGraphical.__init__] self.kernel_size set to: {kernel_size}")
        # self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        print(f"3 [ConvTemporalGraphical.__init__] self.kernel_size set to: {self.kernel_size}")

    def forward(self, x, A):
        print(f"[ConvTemporalGraphical] input x: {x.shape}  | expected in_channels={self.conv.in_channels}")
        print(f"[ConvTemporalGraphical.forward] A.size(0): {A.size(0)} | self.kernel_size: {self.kernel_size}")
        assert A.size(0) == self.kernel_size, f"Expected {self.kernel_size} adjacency matrices, got {A.size(0)}"

        x = self.conv(x)
        print(f"[ConvTemporalGraphical] after conv x: {x.shape}")

        # Fix einsum subscripts: 'k' for spatial kernel
        x = torch.einsum('nctv,kvw->nctw', (x, A))
        print(f"[ConvTemporalGraphical] after einsum x: {x.shape}")

        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False,
                 stride=1, dropout=0, residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        print(f"[st_gcn.__init__] kernel_size: {kernel_size}")
        print(f"[st_gcn.__init__] ConvTemporalGraphical.kernel_size passed: {kernel_size[0]}")
        # kernel_size is (t_kernel_size, num_nodes)
        # assume K=3 as from A_obs.shape[1]
        spatial_kernel_size = 3
        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels, spatial_kernel_size,
            t_kernel_size=kernel_size[0],
            t_padding=(kernel_size[0] - 1) // 2  # <== ADD THIS
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        print(f"[st_gcn] input x: {x.shape}")
        res = self.residual(x)
        print(f"[st_gcn] residual: {res.shape if isinstance(res, torch.Tensor) else 'None'}")
        x, A = self.gcn(x, A)
        print(f"[st_gcn] after gcn x: {x.shape}")
        x = self.tcn(x) + res
        print(f"[st_gcn] after tcn x: {x.shape}")
        if not self.use_mdn:
            x = self.prelu(x)
        return x, A


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(output_feat, output_feat, 3, padding=1))
        for j in range(1, n_txpcnn):
            self.tpcnns.append(nn.Conv2d(output_feat, output_feat, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(output_feat, output_feat, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):
        print(f"[social_stgcnn] initial v: {v.shape}")
        for k in range(self.n_stgcnn):
            print(f"[social_stgcnn] --- st_gcn block {k} ---")
            v, a = self.st_gcns[k](v, a)

        print(f"[social_stgcnn] before tpcnn reshape v: {v.shape}")
        v = v.contiguous().view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        v = v.permute(0, 2, 1, 3)  # (N, C, T, V)
        print(f"[social_stgcnn] reshaped for tpcnn: {v.shape}")

        v = self.prelus[0](self.tpcnns[0](v))
        print(f"[social_stgcnn] after tpcnn[0]: {v.shape}")

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
            print(f"[social_stgcnn] after tpcnn[{k}]: {v.shape}")

        v = self.tpcnn_ouput(v)
        print(f"[social_stgcnn] after tpcnn_output: {v.shape}")

        v = v.permute(0, 2, 1, 3)  # (N, T, C, V)
        v = v.contiguous().view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        print(f"[social_stgcnn] output v shape: {v.shape}")

        return v, a
