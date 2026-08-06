from torch.utils.data import Dataset
import os
from PIL import Image
from utils.utils import*;

# Disable PIL decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform

        self.files = [
            f for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Load another image instead of crashing
            return self.__getitem__((idx + 1) % len(self.files))

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):
    transform_list = []

    if size > 0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = cal_mean_std(style_feat)
    content_mean, content_std = cal_mean_std(content_feat)

    normalized_content_feat = (
        (content_feat - content_mean.expand(size))
        / content_std.expand(size)
    )

    return (
        normalized_content_feat * style_std.expand(size)
        + style_mean.expand(size)
    )


def cal_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4

    batch_size, channels = size[:2]

    feat_mean = (
        feat.view(batch_size, channels, -1)
        .mean(dim=2)
        .view(batch_size, channels, 1, 1)
    )

    feat_var = (
        feat.view(batch_size, channels, -1)
        .var(dim=2, unbiased=False)
        + eps
    )

    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std