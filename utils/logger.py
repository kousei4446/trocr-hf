"""
TensorBoardロギング用モジュール
"""
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


class HTRLogger:
    """HTR学習用のTensorBoardロガー"""

    def __init__(self, log_dir=None, config=None):
        """
        Args:
            log_dir: ログ保存先（Noneなら自動生成）
            config: 設定オブジェクト
        """
        # 日付時刻プレフィックス
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if log_dir is None:
            # 既定: runs/<timestamp>_trocr
            log_dir = os.path.join("runs", f"{timestamp}_trocr")
        else:
            # 指定がある場合も、先頭に日時を付与
            base = os.path.basename(log_dir.rstrip(os.sep))
            parent = os.path.dirname(log_dir.rstrip(os.sep)) or "."
            log_dir = os.path.join(parent, f"{timestamp}_{base}")

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        print("=" * 60)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Start TensorBoard with: python -m tensorboard --logdir=\"{log_dir}\"")
        print("=" * 60)

        # グローバルステップカウンタ
        self.global_step = 0

        # Epoch内の統計用バッファ
        self.reset_epoch_stats()

        # 設定を記録
        if config is not None:
            self.log_hparams(config)

    def reset_epoch_stats(self):
        """エポック統計をリセット"""
        self.epoch_loss_sum = 0.0
        self.epoch_step_count = 0

    def log_hparams(self, config):
        """ハイパーパラメータを記録"""
        # OmegaConfをflatな辞書に変換
        if hasattr(config, "to_container"):
            hparams = OmegaConf.to_container(config, resolve=True)
        else:
            hparams = dict(config)

        # ネストした辞書をフラット化
        flat_hparams = self._flatten_dict(hparams)

        # TensorBoardがサポートする型のみフィルタ
        filtered = {
            k: v for k, v in flat_hparams.items()
            if isinstance(v, (int, float, str, bool))
        }

        self.writer.add_text("config", str(config), 0)

    def _flatten_dict(self, d, parent_key="", sep="/"):
        """ネストした辞書をフラット化"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_step(self, loss, epoch, step):
        """ステップごとのロギング"""
        self.global_step += 1
        self.epoch_loss_sum += loss
        self.epoch_step_count += 1

        # ステップ単位でのlossを記録
        self.writer.add_scalar("train/loss_step", loss, self.global_step)

    def log_epoch(self, epoch, cer, lr=None):
        """エポック終了時のロギング"""
        # エポック平均loss
        avg_loss = self.epoch_loss_sum / max(self.epoch_step_count, 1)

        self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        self.writer.add_scalar("val/CER", cer, epoch)

        if lr is not None:
            self.writer.add_scalar("train/learning_rate", lr, epoch)

        # 次のエポック用にリセット
        self.reset_epoch_stats()

        return avg_loss
    
    def log_test(self,epoch,cer):
        self.writer.add_scalar("test/CER",cer,epoch)

    def close(self):
        """ライターをクローズ"""
        self.writer.close()