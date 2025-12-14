import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from models.connector import Connector
from models.llm import LLM


class TROCRNet(torch.nn.Module):
    """
    TrOCR ベースモデル。config.train.use_llm が true の場合は Connector + GPT-2 を併用。
    """

    def __init__(self, config):
        super().__init__()
        # 学習用ハイパーパラメータを保持（モデル設定とは別物）
        self.hparams = config

        model_name = config.model_name

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.encoder_decoder = VisionEncoderDecoderModel.from_pretrained(model_name)

        # HF モデルの設定を外からも参照しやすいように露出
        self.config = self.encoder_decoder.config
        self.config.pad_token_id = 1
        self.config.eos_token_id = 2
        self.config.decoder_start_token_id = 2

        self.use_llm = config.train.use_llm
        self.connector = None
        self.llm_model = None

        if self.use_llm:
            self.llm_model = LLM(
                model_name=config.train.llm_model_name,
                freeze=True,
            )
            self.connector = Connector(
                input_dim=self.encoder_decoder.config.encoder.hidden_size,
                output_dim=self.llm_model.input_dim,
                downsample=config.train.downsample_connector,
            )

    def forward(self, pixel_values, labels=None):
        
        outputs = self.encoder_decoder(
            pixel_values=pixel_values,
            labels=labels,
        )

        if self.use_llm and self.training and labels is not None:

            # エンコーダ中間出力を LLM 入力空間へ投影して並行で損失を計算
            encoder_outputs = self.encoder_decoder.encoder(pixel_values=pixel_values,output_hidden_states=True)
            hidden_states = encoder_outputs.hidden_states
            
            """
            hidden_states: (B, S, D) 
            small : (12, 578, 384)
            base  : (12, 577, 768)
            """
            llm_losses = {}
            for h in self.hparams.train.selected_layers:
                
                llm_inputs = self.connector(hidden_states[h])
                # llm_inputs.shape -> (B, S, D')：[12, 289, 768]

                # Retokenize labels with GPT-2 tokenizer and align sequence length with LLM inputs
                with torch.no_grad():
                    texts = self.processor.tokenizer.batch_decode(
                        labels.masked_fill(labels == -100, self.config.pad_token_id),
                        skip_special_tokens=True,
                    )
                    tok = self.llm_model.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=llm_inputs.size(1),
                    )
                    gpt2_labels = tok.input_ids.to(llm_inputs.device)
                    gpt2_attn = tok.attention_mask.to(llm_inputs.device)
                    gpt2_labels = gpt2_labels.masked_fill(gpt2_attn == 0, -100)

                llm_outputs = self.llm_model(
                    inputs_embeds=llm_inputs,
                    labels=gpt2_labels,
                    attention_mask=gpt2_attn,
                )
                llm_losses[h] = llm_outputs.loss

            if llm_losses:
                # 全層の平均を代表値として格納し、個別も持たせる
                outputs.llm_loss = torch.stack(list(llm_losses.values())).mean()
                outputs.llm_loss_per_layer = llm_losses

        return outputs

    def generate(self, *args, **kwargs):
        # エンコーダデコーダの generate をそのまま委譲
        return self.encoder_decoder.generate(*args, **kwargs)


def build_model(config):
    """学習スクリプトから呼び出すためのビルダー。"""
    return TROCRNet(config)
