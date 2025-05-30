from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types

class JointRandomScale(object):
    """
    Rescales the input PIL.Images to a randomly chosen size between two thresholds.
    
    The size of the smaller edge will be scaled to 'chosen_size'.
    Different interpolation methods are applied to different images:
      - BILINEAR for imgs[0]
      - NEAREST for imgs[1]
    
    Attributes:
        min_size (int): Minimum threshold value for resizing.
        max_size (int): Maximum threshold value for resizing.
    """
    
    def __init__(self, min_size, max_size): 
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs):
        self.chosen_size = random.randint(self.min_size, self.max_size)
        w, h = imgs[0].size
        if (w <= h and w == self.chosen_size) or (h <= w and h == self.chosen_size):
            return imgs
        
        if w < h:
            ow = self.chosen_size
            oh = int(self.chosen_size * h / w)
        else:
            oh = self.chosen_size
            ow = int(self.chosen_size * w / h)
        
        resized_imgs = [
            imgs[0].resize((ow, oh), Image.BILINEAR),
            imgs[1].resize((ow, oh), Image.NEAREST)
        ]
        
        return resized_imgs


class JointScale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation="BILINEAR"):
        self.size = size
        self.interpolation = getattr(Image, interpolation)

    def __call__(self, imgs):
        w, h = imgs[0].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return imgs
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return [img.resize((ow, oh), self.interpolation) for img in imgs]
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return [img.resize((ow, oh), self.interpolation) for img in imgs]


class JointCenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        w, h = imgs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointPad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, imgs):
        return [ImageOps.expand(img, border=self.padding, fill=self.fill) for img in imgs]


class JointLambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, imgs):
        return [self.lambd(img) for img in imgs]


class JointRandomCrop(object):
    """Crops the given list of PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]

class FixedUpperLeftCrop(object):
    """Crops the given list of PIL.Image starting from the upper left corner
    to have a region of the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        th, tw = self.size  # th for target height, tw for target width

        return [img.crop((0, 0, tw, th)) for img in imgs]  # Crop from the top-left corner

class JointRandomRotate90(object):
    """Randomly rotates the given list of PIL.Image by 90 degrees with a probability of 0.5
    """

    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.ROTATE_90) for img in imgs]
        return imgs

class JointRandomHorizontalFlip(object):
    """Randomly horizontally flips the given list of PIL.Image with a probability of 0.5
    """

    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs
    
class JointRandomFlip(object):
    """Randomly  flips the given list of PIL.Image
    """

    def __call__(self, imgs):
        r = random.random()
        if r < .25:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        elif r < .5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]
        elif r < .75:
            return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]

        return imgs


class JointRandomSizedCrop(object):
    """Random crop the given list of PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                assert(imgs[0].size == (w, h))

                return [img.resize((self.size, self.size), self.interpolation) for img in imgs]

        # Fallback
        scale = JointScale(self.size, interpolation=self.interpolation)
        crop = JointCenterCrop(self.size)
        return crop(scale(imgs))
