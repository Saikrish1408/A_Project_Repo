from __future__ import print_function
import cv2 as cv
import numpy as np
from datetime import datetime
import os

def image_enhance(input_image_path):

    if not os.path.exists(input_image_path):
        print(f'Could not find the image: {input_image_path}')
        return -1


    image = cv.imread(input_image_path)
    if image is None:
        print(f'Could not open or find the image: {input_image_path}')
        return -1

    new_image = np.zeros(image.shape, image.dtype)


    alpha = 3.0  
    beta = 70    # Brightness control

    # Apply the brightness 
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_image_path = os.path.join('static/uploads', f'enhanced_img.png')


    cv.imwrite(output_image_path, new_image)

    return output_image_path
