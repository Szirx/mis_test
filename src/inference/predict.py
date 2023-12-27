import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
import cv2
import albumentations as A
import os


def predict_image(model, image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Resize(384, 480, interpolation=cv2.INTER_NEAREST)
    aug = transform(image=image)
    image = Image.fromarray(aug['image'])

    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    device = 'cpu'
    model.to(device)
    image = image.to(device)

    with torch.no_grad():

        image = image.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


if __name__ == '__main__':

    PATH_TO_IMAGES_DIR = '../../images'
    PATH_TO_MODEL = '../../models/Unet-Mobilenet_v2_best.pt'
    PATH_TO_PREDICTED = 'predicted/predicted_mask.png'
    files = os.listdir(PATH_TO_IMAGES_DIR)
    PATH_TO_IMAGE = 'images/' + files[0]  # в папке должно находится одно изображение(берет первое из списка)

    inference_model = torch.load(PATH_TO_MODEL, map_location='cpu')
    predicted_mask = predict_image(inference_model, PATH_TO_IMAGE)

    image_array = np.array(predicted_mask)
    cv2.imwrite(PATH_TO_PREDICTED, image_array)
