import albumentations as A
import cv2

t_train = A.Compose([A.Resize(384, 480, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                     A.GaussNoise()])

t_val = A.Compose([A.Resize(384, 480, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])
