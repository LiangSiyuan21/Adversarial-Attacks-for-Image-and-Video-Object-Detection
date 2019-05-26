import cv2
import torch
from data.dataset import inverse_normalize
from utils import array_tool as at

target_img = torch.randn((1, 3, 400, 400))

target_img = torch.clamp(target_img, -1.0, 1.0)

target_img_ = inverse_normalize(at.tonumpy(target_img[0]))

img = cv2.imwrite('target_img.jpg', target_img_.transpose((1, 2, 0)))

# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
