import numpy as np
from torchvision.transforms import v2

# Custom transform to crop image based on the bounding box
class CropToBBox:
    def __call__(self, image, bboxes, class_labels):
        
        # Assuming we're cropping to the first bounding box
        bbox = bboxes[0].tolist()  # Get the first bounding box (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Crop the image using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        transformed_target = {}
        transformed_target["image"] = cropped_image
        transformed_target["bboxes"] = bboxes
        transformed_target["class_labels"] = class_labels

        return transformed_target