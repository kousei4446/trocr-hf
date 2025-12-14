# utils/transforms.py
from PIL import Image
from utils.augmentations import (
    Compose,
    RandomRotate,
    RandomGaussianBlur,
    RandomDilation,
    RandomErosion,
    RandomDownsample,
    RandomUnderline,
)

def build_train_transform(aug_cfg):
    if not aug_cfg or not getattr(aug_cfg, "enable", False):
        return None

    resample_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }

    transforms = []

    rotate_cfg = getattr(aug_cfg, "rotate", None)
    if rotate_cfg:
        transforms.append(
            RandomRotate(
                degrees=getattr(rotate_cfg, "degrees", 10.0),
                p=getattr(rotate_cfg, "p", 0.5),
                fill=getattr(rotate_cfg, "fill", 255),
            )
        )

    blur_cfg = getattr(aug_cfg, "gaussian_blur", None)
    if blur_cfg:
        transforms.append(
            RandomGaussianBlur(
                radius_range=(
                    getattr(blur_cfg, "radius_min", 0.5),
                    getattr(blur_cfg, "radius_max", 1.2),
                ),
                p=getattr(blur_cfg, "p", 0.3),
            )
        )

    dil_cfg = getattr(aug_cfg, "dilation", None)
    if dil_cfg:
        transforms.append(
            RandomDilation(
                size=getattr(dil_cfg, "size", 3),
                p=getattr(dil_cfg, "p", 0.3),
            )
        )

    ero_cfg = getattr(aug_cfg, "erosion", None)
    if ero_cfg:
        transforms.append(
            RandomErosion(
                size=getattr(ero_cfg, "size", 3),
                p=getattr(ero_cfg, "p", 0.3),
            )
        )

    down_cfg = getattr(aug_cfg, "downsample", None)
    if down_cfg:
        up_name = getattr(down_cfg, "up_interpolation", "bicubic")
        up_mode = resample_map.get(str(up_name).lower(), Image.BICUBIC)
        transforms.append(
            RandomDownsample(
                ratio_range=(
                    getattr(down_cfg, "ratio_min", 0.5),
                    getattr(down_cfg, "ratio_max", 0.8),
                ),
                p=getattr(down_cfg, "p", 0.4),
                up_interpolation=up_mode,
            )
        )

    ul_cfg = getattr(aug_cfg, "underline", None)
    if ul_cfg:
        transforms.append(
            RandomUnderline(
                p=getattr(ul_cfg, "p", 0.3),
                thickness_range=(
                    getattr(ul_cfg, "thickness_min", 1),
                    getattr(ul_cfg, "thickness_max", 3),
                ),
                offset_range=(
                    getattr(ul_cfg, "offset_min", 1),
                    getattr(ul_cfg, "offset_max", 5),
                ),
                color=tuple(getattr(ul_cfg, "color", (0, 0, 0))),
            )
        )

    return Compose(transforms) if transforms else None
