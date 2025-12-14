from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import torch.nn as nn

class LLMwithGPT2(nn.Module):
    """
    GPT-2ベースの大規模言語モデル。
    - 入力: (B, seq_len, D=input_dim)
    - 出力: (B, seq_len, vocab_size)
    """
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.input_dim = self.model.config.n_embd
        self.config = self.model.config
        self.model.requires_grad_(False)
        
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            x (torch.Tensor): (B, seq_len, D=input_dim)
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
            use_cache=False,
            return_dict=True,
        )

        return outputs