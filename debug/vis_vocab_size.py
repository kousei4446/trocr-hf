from transformers import TrOCRProcessor
import sys
from omegaconf import OmegaConf

def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf

config = parse_args()    
processor = TrOCRProcessor.from_pretrained(config.model_name)

print("Vocab size:", processor.tokenizer.vocab_size)


id2token = {v: k for k, v in processor.tokenizer.get_vocab().items()}
single_char_tokens = []

for token, idx in processor.tokenizer.get_vocab().items():
    # 特殊トークンを除外
    if token.startswith("<") and token.endswith(">"):
        continue

    # Unicode 1文字かどうか
    if len(token) == 1:
        single_char_tokens.append(token)

print("1文字トークン一覧:", single_char_tokens)
print("数:", len(single_char_tokens))
