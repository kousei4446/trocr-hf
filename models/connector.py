import torch.nn as nn
import torch.nn.functional as F


class Connector(nn.Module):
    """
    Projects encoder hidden states into the LLM embedding space and optionally downsamples time steps.
    Input:  (B, seq_len, input_dim)
    Output: (B, seq_len / downsample, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, downsample: int):
        super().__init__()
        self.downsample = max(int(downsample), 1)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, seq_len, input_dim)
        """
        out = self.input_projection(x)
        if self.downsample > 1:
            # 時系列方向にダウンサンプリング
            out = F.max_pool1d(
                out.transpose(1, 2), kernel_size=self.downsample, stride=self.downsample
            ).transpose(1, 2)
        return out
