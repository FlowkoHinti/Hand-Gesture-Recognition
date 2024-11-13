import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
import torch

# Custom transform to crop image based on the bounding box
class CropToBBox:
    def __call__(self, image, bboxes, class_labels):
        
        # Assuming we're cropping to the first bounding box
        bbox = bboxes[0].tolist()  # Get the first bounding box (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Crop the image using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        # To PIL Image (required for torchvision transforms)
        cropped_image = v2.ToImage()(cropped_image)

        # scale the image to 224x224
        cropped_image = v2.Resize((224, 224))(cropped_image)

        cropped_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(cropped_image)

        cropped_image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_image)

        transformed_target = {}
        transformed_target["image"] = cropped_image
        transformed_target["bboxes"] = bboxes[0]                #only the first bounding box
        transformed_target["class_labels"] = class_labels[0]    #only the first class label

        # Add Resnet/Vit Transforms

        return transformed_target
    
# for usage
class BBoxtoImageNet:
    def __call__(self, image, x1, y1, x2, y2):
        # To PIL Image (required for torchvision transforms)
        cropped_image = v2.ToImage()(image)

        # Crop the image using the bounding box
        width = x2 - x1
        height = y2 - y1
        cropped_image = crop(image, top=y1, left=x1, height=height, width=width)

        # scale the image to 224x224
        cropped_image = v2.Resize((224, 224))(cropped_image)

        cropped_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(cropped_image)

        cropped_image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_image)

        return cropped_image.unsqueeze(int(0))