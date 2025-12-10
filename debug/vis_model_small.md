# VisionEncoderDecoderModelの構成
### small:61596672 parameters

```
Model 'VisionEncoderDecoderModel(
  (encoder): DeiTModel(
    (embeddings): DeiTEmbeddings(
      (patch_embeddings): DeiTPatchEmbeddings(
        (projection): Conv2d(3, 384, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): DeiTEncoder(
      (layer): ModuleList(
        (0-11): 12 x DeiTLayer(
          (attention): DeiTAttention(
            (attention): DeiTSelfAttention(
              (query): Linear(in_features=384, out_features=384, bias=True)
              (key): Linear(in_features=384, out_features=384, bias=True)
              (value): Linear(in_features=384, out_features=384, bias=True)
            )
            (output): DeiTSelfOutput(
              (dense): Linear(in_features=384, out_features=384, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): DeiTIntermediate(
            (dense): Linear(in_features=384, out_features=1536, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): DeiTOutput(
            (dense): Linear(in_features=1536, out_features=384, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
    (pooler): DeiTPooler(
      (dense): Linear(in_features=384, out_features=384, bias=True)
      (activation): Tanh()
    )
  )
  (decoder): TrOCRForCausalLM(
    (model): TrOCRDecoderWrapper(
      (decoder): TrOCRDecoder(
        (embed_tokens): TrOCRScaledWordEmbedding(64044, 256, padding_idx=1)
        (embed_positions): TrOCRLearnedPositionalEmbedding(514, 256)
        (layernorm_embedding): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0-5): 6 x TrOCRDecoderLayer(
            (self_attn): TrOCRAttention(
              (k_proj): Linear(in_features=256, out_features=256, bias=True)
              (v_proj): Linear(in_features=256, out_features=256, bias=True)
              (q_proj): Linear(in_features=256, out_features=256, bias=True)
              (out_proj): Linear(in_features=256, out_features=256, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): TrOCRAttention(
              (k_proj): Linear(in_features=384, out_features=256, bias=True)
              (v_proj): Linear(in_features=384, out_features=256, bias=True)
              (q_proj): Linear(in_features=256, out_features=256, bias=True)
              (out_proj): Linear(in_features=256, out_features=256, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (final_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (output_projection): Linear(in_features=256, out_features=64044, bias=False)
  )
)' loaded.
```