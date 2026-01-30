from __future__ import division
import math
import random
from PIL import Image, ImageOps
import numbers
import types

# joint data-augmentation transforms for paired PIL images (image + mask).

class JointScale:
    """
    Rescales a pair of PIL.Images (image and mask) to a target size.
    
    You can specify either both width and height, or just one of them (proportional resize).
    Applies:
        - BILINEAR interpolation for the first image (e.g. RGB image)
        - NEAREST interpolation for the second image (e.g. segmentation mask)
    """

    def __init__(self, w=None, h=None):
        if h is None and w is None:
            raise ValueError("At least one of h or w must be specified.")
        self.target_h = h
        self.target_w = w

    def __call__(self, imgs):
        img = imgs[0]
        w, h = img.size  # PIL: size = (width, height)

        # Compute target size if only one dimension is provided
        if self.target_h is None:
            target_w = self.target_w
            target_h = int(target_w * h / w)
        elif self.target_w is None:
            target_h = self.target_h
            target_w = int(target_h * w / h)
        else:
            target_h = self.target_h
            target_w = self.target_w

        resized_imgs = [
            imgs[0].resize((target_w, target_h), Image.BILINEAR),
            imgs[1].resize((target_w, target_h), Image.NEAREST)
        ]
        return resized_imgs
    
class JointRandomScale:
    """
    Rescales input PIL.Images to a randomly chosen height
    between min_size and max_size. Width is adjusted to preserve aspect ratio.

    Uses JointScale internally to perform the resize.
    """

    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs):
        chosen_height = random.randint(self.min_size, self.max_size)
        scale = JointScale(h=chosen_height)
        return scale(imgs)

class JointRotate:
    """
    Rotates a pair of PIL.Images (image and mask) by the specified angle.
    Applies:
        - BILINEAR interpolation for the first image (e.g. RGB image)
        - NEAREST interpolation for the second image (e.g. segmentation mask)
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, imgs):
        rotated_imgs = [
            imgs[0].rotate(self.angle, resample=Image.BILINEAR, expand=True),
            imgs[1].rotate(self.angle, resample=Image.NEAREST, expand=True)
        ]
        return rotated_imgs
    
class JointRandomRotate:
    """
    Rotates a pair of PIL.Images (image and mask) by a random angle between min_angle and max_angle.
    Uses JointRotate internally.
    """

    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, imgs):
        angle = random.uniform(self.min_angle, self.max_angle)
        return JointRotate(angle)(imgs)


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



class JointRandomRotate90(object):
    """Randomly rotates the given list of PIL.Image by 90 degrees with a probability of 0.5
    """

    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.ROTATE_90) for img in imgs]
        return imgs

class JointRandomRotateStep90(object):
    def __init__(self):
        self.rotation_map = {
            0: None,
            90: Image.ROTATE_90,
            180: Image.ROTATE_180,
            270: Image.ROTATE_270,
        }

    def __call__(self, imgs):
        angle = random.choice([0, 90, 180, 270])
        if self.rotation_map[angle] is None:
            return imgs
        return [img.transpose(self.rotation_map[angle]) for img in imgs]

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
