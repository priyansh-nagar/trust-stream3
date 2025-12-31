import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def cam_heatmap(model, tensor, image):
    cam = GradCAM(model=model, target_layers=[model.blocks[-1]])
    grayscale = cam(input_tensor=tensor)[0]
    img = np.array(image).astype(float)/255
    return show_cam_on_image(img, grayscale, use_rgb=True)