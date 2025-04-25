import random
from torchvision.transforms import functional as F

class RandomDiscreteRotation:
    def __init__(self, angles, interpolation=0):
        self.angles = angles
        self.interpolation = interpolation

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle, interpolation=self.interpolation)