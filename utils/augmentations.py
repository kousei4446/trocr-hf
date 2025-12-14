import random
from typing import Callable, Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter


class Compose:
    """Apply a sequence of transforms to a PIL image."""

    def __init__(self, transforms: Iterable[Callable[[Image.Image], Image.Image]]):
        self.transforms = list(transforms)

    def __call__(self, img: Image.Image) -> Image.Image:
        for t in self.transforms:
            if t is not None:
                img = t(img)
        return img


class RandomRotate:
    def __init__(self, degrees: float = 10.0, p: float = 0.5, fill: int = 255):
        self.degrees = degrees
        self.p = p
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=self.fill)


class RandomGaussianBlur:
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 1.2), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomDilation:
    def __init__(self, size: int = 3, p: float = 0.3):
        self.size = size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.MaxFilter(self.size))


class RandomErosion:
    def __init__(self, size: int = 3, p: float = 0.3):
        self.size = size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.MinFilter(self.size))


class RandomDownsample:
    def __init__(
        self,
        ratio_range: Tuple[float, float] = (0.5, 0.8),
        p: float = 0.4,
        up_interpolation: int = Image.BICUBIC,
    ):
        self.ratio_range = ratio_range
        self.p = p
        self.up_interpolation = up_interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        ratio = random.uniform(*self.ratio_range)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        if new_w == w and new_h == h:
            return img
        down = img.resize((new_w, new_h), resample=Image.BILINEAR)
        return down.resize((w, h), resample=self.up_interpolation)


class RandomUnderline:
    def __init__(
        self,
        p: float = 0.3,
        thickness_range: Tuple[int, int] = (1, 3),
        offset_range: Tuple[int, int] = (1, 5),
        color: Sequence[int] = (0, 0, 0),
    ):
        self.p = p
        self.thickness_range = thickness_range
        self.offset_range = offset_range
        self.color = tuple(color)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        offset = random.randint(*self.offset_range)
        y = max(0, min(h - 1, h - offset))
        thickness = random.randint(*self.thickness_range)
        draw = ImageDraw.Draw(img)
        for t in range(thickness):
            y_t = max(0, min(h - 1, y - t))
            draw.line([(0, y_t), (w, y_t)], fill=self.color, width=1)
        return img
