import os
import sys
from pathlib import Path
from typing import Dict

import torch
from PIL import Image as PILImage
from omegaconf import OmegaConf
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import xlsxwriter

from utils.excels import DiffHighlighter

highlighter = DiffHighlighter()


def write_three_lines_rich(
    workbook,
    worksheet,
    row: int,
    col: int,
    gt_text: str,
    pred_base: str,
    pred_ft: str,
    mask_gt,
    mask_base,
    mask_ft,
):
    """
    1セルに
      GT: xxx
      BASE: yyy
      FT: zzz
    を改行でまとめ、間違い文字だけ赤文字にする
    """
    fmt_normal = workbook.add_format({})
    fmt_red    = workbook.add_format({"font_color": "red", "bold": True})
    fmt_label  = workbook.add_format({"bold": True})       # 「GT: 」などラベル用
    fmt_cell   = workbook.add_format({"text_wrap": True})  # セル内改行有効化

    pieces = []

    def add_line(label: str, text: str, mask):
        # ラベル部分（例: "GT: "）
        pieces.append(fmt_label)
        pieces.append(label)

        if not text:
            # 空文字ならラベルだけで終わり
            return

        # マスクに従って、連続した文字をまとめてフォーマット付け
        current_flag = mask[0]
        buf = []

        for ch, flag in zip(text, mask):
            if flag == current_flag:
                buf.append(ch)
            else:
                fmt = fmt_red if current_flag else fmt_normal
                pieces.append(fmt)
                pieces.append("".join(buf))
                buf = [ch]
                current_flag = flag

        # 最後のバッファ
        if buf:
            fmt = fmt_red if current_flag else fmt_normal
            pieces.append(fmt)
            pieces.append("".join(buf))

    # 1行目 GT
    add_line("GT: ", gt_text, mask_gt)
    pieces.append("\n")

    # 2行目 BASE
    add_line("BS: ", pred_base, mask_base)
    pieces.append("\n")

    # 3行目 FT（最後は改行なし）
    add_line("FT: ", pred_ft, mask_ft)

    # 1セルにリッチテキストとして書き込み
    worksheet.write_rich_string(row, col, *pieces, fmt_cell)


def generate_text(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    img: PILImage.Image,
    device: torch.device,
    gen_kwargs,
) -> str:
    with torch.no_grad():
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values, **gen_kwargs)
        sequences = out.sequences if hasattr(out, "sequences") else out
        sequences = sequences.detach().cpu()
    return processor.batch_decode(sequences, skip_special_tokens=True)[0]


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


def load_model(model_path: str, device: torch.device):
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()
    return processor, model


def build_gen_kwargs(model: VisionEncoderDecoderModel, max_new_tokens: int = 64):
    cfg = model.config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "pad_token_id": getattr(cfg, "pad_token_id", None),
        "decoder_start_token_id": getattr(cfg, "decoder_start_token_id", None),
    }
    # None のものは除外
    return {k: v for k, v in gen_kwargs.items() if v is not None}


def load_labels(labels_path: str) -> Dict[str, str]:
    """labels.txt を読み込んで {image_id_without_ext -> text} の dict を返す"""
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(None, 1)
            fname = parts[0]
            image_id = os.path.splitext(fname)[0]  # ファイル名から拡張子を除いたものを ID にする
            text = parts[1] if len(parts) > 1 else ""
            labels[image_id] = text
    return labels


# Excel に保存する内容を生成する関数（xlsxwriter版）
def export_to_excel(
    img_path: Path,
    gt_text: str,
    pred_base: str,
    pred_ft: str,
    workbook,
    worksheet,
    row: int,
):
    # 画像を挿入（A列=col 0）
    # row, col は 0 始まり
    worksheet.insert_image(row, 0, str(img_path), {
        "x_scale": 0.5,  # 必要に応じて調整
        "y_scale": 0.5,
    })

    # 差分マスク作成（GT / BASE / FT）
    mask_gt, mask_base, mask_ft = highlighter.make_diff_masks(gt_text, pred_base, pred_ft)

    # テキストは B列（col=1）の1セルに3行まとめて書く
    write_three_lines_rich(
        workbook,
        worksheet,
        row=row,
        col=1,  # B列
        gt_text=gt_text,
        pred_base=pred_base,
        pred_ft=pred_ft,
        mask_gt=mask_gt,
        mask_base=mask_base,
        mask_ft=mask_ft,
    )


def main():
    config = parse_args()
    device = config.device if torch.cuda.is_available() else "cpu"

    image_dir = Path(config.excel.images_dir)
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    labels_path = Path(config.excel.labels_path)
    labels = load_labels(labels_path)

    ft_processor, ft_model = load_model(config.excel.ft_ckpt, device)
    base_processor, base_model = load_model(config.model_name, device)
    ft_gen_kwargs = build_gen_kwargs(ft_model, 64)
    base_gen_kwargs = build_gen_kwargs(base_model, 64)

    out_path = Path(config.excel.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== ここから xlsxwriter =====
    workbook = xlsxwriter.Workbook(str(out_path))
    worksheet = workbook.add_worksheet()

    # 列幅調整（A=画像, B=テキスト）
    worksheet.set_column("A:A", 20)
    worksheet.set_column("B:B", 60)

    row = 0  # 0 始まり

    for path in image_paths:
        with PILImage.open(path) as pil:
            img = pil.convert("RGB")

        pred_ft = generate_text(ft_processor, ft_model, img, device, ft_gen_kwargs)
        pred_base = generate_text(base_processor, base_model, img, device, base_gen_kwargs)

        gt_text = labels.get(path.stem, "")

        export_to_excel(
            img_path=path,
            gt_text=gt_text,
            pred_base=pred_base,
            pred_ft=pred_ft,
            workbook=workbook,
            worksheet=worksheet,
            row=row,
        )

        print(f"Image: {path.name}")
        print(f"GT Prediction:  {gt_text}")
        print(f"Base Prediction:{pred_base}")
        print(f"FT Prediction:  {pred_ft}")
        print("-" * 100)

        row += 1  # 1画像につき1行（B列は内部で3行表示される）

    workbook.close()


if __name__ == "__main__":
    main()
