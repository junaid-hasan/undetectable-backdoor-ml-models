import torch
import random
import math

def pixel_function(i, j, type):
    if type:
        return 1/2 * abs(math.sin(i) + math.cos(j))
    return 1/2 * abs(math.sin(j) - math.cos(i))

def generate_data(num_of_data: int, image_size: tuple[int, int]):
    images = []
    labels = []
    for i in range(num_of_data):
        type0 = random.uniform(0, 1) < 1/2
        image = torch.empty(image_size[0], image_size[1])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = pixel_function(i, j, type0) + random.uniform(-0.25, 0.25)
        
        images.append(image)
        labels.append(int(type0))

    return images, labels