from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_MODEL_NAME = "microsoft/trocr-small-handwritten"


@dataclass
class TrocrConfig:
    """Lightweight config for building a TrOCR model using HF weights."""

    model_name: str = DEFAULT_MODEL_NAME
    vocab_size: Optional[int] = None
    freeze_encoder: bool = True
    train_last_encoder_layer: bool = True
    train_decoder_layers: int = 2  # number of decoder layers to keep trainable from the top


class TrocrFactory:
    @staticmethod
    def load(config: TrocrConfig) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
        """
        Load a TrOCR model/processor from Hugging Face weights and optionally
        adapt training scope (freeze encoder, keep top decoder layers trainable,
        and resize vocab if needed).
        """

        processor = TrOCRProcessor.from_pretrained(config.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(config.model_name)

        tokenizer = processor.tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id or model.config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id or model.config.eos_token_id
        # Prefer BOS if available; fall back to EOS for decoder start.
        model.config.decoder_start_token_id = (
            getattr(tokenizer, "bos_token_id", None)
            or model.config.decoder_start_token_id
            or model.config.eos_token_id
        )

        # If you introduce a custom vocab, resize decoder embeddings/head.
        if config.vocab_size and config.vocab_size != model.config.vocab_size:
            model.decoder.resize_token_embeddings(config.vocab_size)
            model.config.vocab_size = config.vocab_size
            if hasattr(model, "tie_weights"):
                model.tie_weights()

        if config.freeze_encoder:
            TrocrFactory._freeze_all(model.encoder)
            if config.train_last_encoder_layer:
                TrocrFactory._unfreeze_last_encoder_block(model)

        if config.train_decoder_layers is not None:
            TrocrFactory._selective_decoder_training(model, config.train_decoder_layers)

        return model, processor

    @staticmethod
    def _freeze_all(module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze_last_encoder_block(model) -> None:
        encoder_layers = getattr(model.encoder, "encoder", None)
        encoder_layers = getattr(encoder_layers, "layer", None)
        if encoder_layers:
            for p in encoder_layers[-1].parameters():
                p.requires_grad = True

    @staticmethod
    def _selective_decoder_training(model, train_decoder_layers: int) -> None:
        decoder_layers = model.decoder.model.decoder.layers
        if train_decoder_layers <= 0:
            TrocrFactory._freeze_all(model.decoder)
            return

        # Freeze lower decoder blocks, keep the top N trainable.
        for layer in decoder_layers[:-train_decoder_layers]:
            TrocrFactory._freeze_all(layer)
        for layer in decoder_layers[-train_decoder_layers:]:
            for p in layer.parameters():
                p.requires_grad = True


def load_small_handwritten(**kwargs):
    """
    Convenience helper to load `microsoft/trocr-small-handwritten` weights and
    return `(model, processor)`. Extra keyword args are forwarded to TrocrConfig.
    """

    config = TrocrConfig(**kwargs)
    return TrocrFactory.load(config)