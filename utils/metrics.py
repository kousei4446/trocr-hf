import re

import editdistance

import nltk
# quiet=True 縺ｧ繝繧ｦ繝ｳ繝ｭ繝ｼ繝峨Γ繝・そ繝ｼ繧ｸ繧呈椛蛻ｶ
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize


def normalize_36_charset(text: str, keep_space: bool = False) -> str:
    """
    IAM の 36 文字（小文字英数字＋スペース）のみを残す正規化。
    keep_space=False にするとスペースも除外して全て連結する。
    """
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789" + (" " if keep_space else "")
    pattern = re.compile(f"[^{re.escape(allowed)}]+")

    text = text.lower()
    text = pattern.sub(" ", text if keep_space else text.replace(" ", ""))
    text = re.sub(r"\s+", " ", text).strip() if keep_space else text.strip()
    return text


class CER:
    """Character Error Rate"""

    def __init__(self, normalize_fn=None):
        self.total_dist = 0
        self.total_len = 0
        self.normalize_fn = normalize_fn or (lambda x: x)
        
    def update(self, prediction, target):
        prediction = self.normalize_fn(prediction)
        target = self.normalize_fn(target)
        # skip empty targets to avoid zero-length divisions
        if len(target) == 0:
            print("Warning: empty target encountered in CER calculation; skipping.")
            return
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)

    def score(self):
        return self.total_dist / self.total_len if self.total_len > 0 else 0.0

    def reset(self):
        self.total_dist = 0
        self.total_len = 0
