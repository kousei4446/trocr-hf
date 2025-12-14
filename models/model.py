import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from models.connector import Connector
from models.llm import LLMwithGPT2

class TROCRNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__(config)
        
        self.processor = TrOCRProcessor.from_pretrained(config.model_name)
        self.encoder_decoder = VisionEncoderDecoderModel.from_pretrained(config.encoder.model_name)
        
        # LLMの入力次元に合わせるコネクターモジュール
        self.connector = Connector(
            input_dim=self.encoder.config.hidden_size,
            output_dim=llm_model.input_dim,
        )
        
        llm_model = LLMwithGPT2()
        self.llm_model = llm_model

