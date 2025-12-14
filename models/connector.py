import torch.nn as nn

class Connector:
    """
    LLMへの入力を変換するコネクターモジュールの基底クラス。
    - 出力: (B, seq_len, D=output_dim)
    """
    def __init__(self, input_dim: int,output_dim: int):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        # シーケンス長を半分にダウンサンプリングする層
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, seq_len, D=input_dim)
        """
        out = self.input_projection(x)
        out = out.transpose(1, 2)  # (B, D, seq_len)
        out = self.downsample(out)  # (B, D, seq_len/2
        out = out.transpose(1, 2)  # (B, seq_len/2, D)
        
        return out
