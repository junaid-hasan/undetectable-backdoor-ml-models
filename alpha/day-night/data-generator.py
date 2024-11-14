import torch
import random

def generate_data(num_of_data: int, image_size: tuple[int, int]):
    """
    Generates `num_of_data` images of dimension `image_size` along with a
    label for whether the image is of daytime or nighttime.

    About half the images will be day images, about half will be night images.
    
    Day image:
        -labeled 1
        -pixels average near 0.7
    Night image:
        -labeled -1
        -pixels average near 0.3
    """
    images = []
    labels = []
    for i in range(num_of_data):
        if random.uniform(0, 1) < 1/2:
            avg_brightness = 0.7
            is_day = 1
        else:
            avg_brightness = 0.3
            is_day = -1
        
        image = torch.empty(image_size[0], image_size[1]).uniform_(
            avg_brightness - 0.2, avg_brightness + 0.2
        )
        
        images.append(image)
        labels.append(is_day)

    return images, labels

data = generate_data(1000, (16,16))