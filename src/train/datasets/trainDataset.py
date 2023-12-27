from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch
from torchvision import transforms as T


class TrainDataset(Dataset):

    def __init__(self, img_path, mask_path, x, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.x = x
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.x[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.x[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        return img, mask
