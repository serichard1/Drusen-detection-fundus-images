import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from transformers.models.segformer.modeling_segformer import *
import glob
from torchvision.datasets.folder import default_loader

class Resnet18(nn.Module):
    def __init__(self, weights="ResNet18_Weights.DEFAULT", pretrained=True):
        super().__init__()
        self.name = "Resnet18"
        self.net = models.resnet18(weights=weights)
        self.num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(self.num_ftrs, 1)
        self.activ = nn.Sigmoid()
        if pretrained:
            print(f"loading pretrained weights from: results/pretrained_Resnet18.pt")
            try:
                self.load_state_dict(torch.load(f"results/pretrained_Resnet18.pt"))
            except:
                print("couldn't find suitable pretrained weights for this model; training from ImageNet weights")

    def forward(self, x):
        return self.activ(self.net(x))


class Efficient_b2(nn.Module):
    def __init__(self, weights="EfficientNet_B2_Weights.IMAGENET1K_V1", pretrained=True):
        super().__init__()
        self.name = "Efficient_b2"
        net = models.efficientnet_b2(weights=weights)
        net.classifier[1] = nn.Linear(1408, 1)
        self.base_model = net
        self.sigm = nn.Sigmoid()
        if pretrained:
            print(f"loading pretrained weights from: results/pretrained_Efficient_b2.pt")
            try:
                self.load_state_dict(torch.load(f"results/pretrained_Efficient_b2.pt"))
            except:
                print("couldn't find suitable pretrained weights for this model; training from ImageNet weights")

    def forward(self, x):
        return self.sigm(self.base_model(x))

 
class EnsembleModel(nn.Module):
    def __init__(self, pretrained=False):
        super(EnsembleModel, self).__init__()
        self.name = "EnsembleModel"
        self.modelA = Resnet18()
        self.modelB = Efficient_b2()
        
        self.modelA.net.fc = nn.Linear(in_features=512, out_features=32, bias=True)
        self.modelB.base_model.classifier[1] = nn.Linear(in_features=1408, out_features=64, bias=True)
        self.modelA.activ = nn.ReLU()
        self.modelB.sigm = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.activ = nn.Sigmoid()
        self.fc1 = nn.Linear(32+64, 1)
        if pretrained:
            print(f"loading pretrained weights from: results/pretrained_EnsembleModel.pt")
            self.load_state_dict(torch.load(f"results/pretrained_EnsembleModel.pt"))
        
    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        
        x = torch.cat((out1, out2), dim=1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.activ(x)
        return x

class DataAugmentation:
    """
    Parameters
    ----------
    resize : int
        desired width and height of the image
    augmentation_type: str
        augmentations to be executed;
        "TrivialAugmentWide" => Pytorch implementation from https://github.com/automl/trivialaugment
        "albumentations" => Python fast and efficient library from https://github.com/albumentations-team/albumentations
        "ietk" => retinal image specific augmentations from https://github.com/adgaudio/ietk-ret
    """
    def __init__(
        self,
        resize=420,
        augmentation_type = "standard"
    ):
        self.resize = resize
        self.augmentation_types = ["TrivialAugmentWide","albumentations","ietk","standard"]
        self.augmentation_type = augmentation_type
        if self.augmentation_type not in self.augmentation_types:
            raise ValueError("Invalid augmentation_type. Expected one of : %s" % self.augmentation_types)

        if self.augmentation_type == "TrivialAugmentWide":
            self.transformations = transforms.Compose(
            [
                transforms.Resize((resize,resize)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
            ]
        )
        elif self.augmentation_type == "albumentations":
            self.transformations = A.Compose(
                [
                    A.GridDistortion(p=0.2),
                    A.CLAHE(p=0.1),
                    A.GaussianBlur(p=0.5),
                    A.Resize(height=self.resize,width=self.resize),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    ToTensorV2()
                ]
            )
        elif self.augmentation_type == "ietk":
            methods = ["sA+sC+sX+sZ, sA+sB+sC+sW+sX, A+B+C+X, sC+sX"]
            self.transformations = transforms.Compose(
                [
                    # Ietk_transform(choice(methods)),
                    transforms.Resize((resize,resize)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transformations = transforms.Compose(
                [
                    transforms.Resize((resize,resize)),
                    # transforms.RandomHorizontalFlip(p=0.2),
                    # transforms.RandomVerticalFlip(p=0.2),
                    transforms.ToTensor(),
                ]
            )
            
    def __call__(self, image):
        """
        Apply transformation.
        Parameters
        ----------
        img : numpy array
            input image.

        Returns
        -------
        img: torch.Tensor
            augmented image
        """
        if self.augmentation_type == "albumentations":
            return self.transformations(image=image)['image']
        else:
            return self.transformations(image)
        

class InferenceBasic(Dataset):
    def __init__(self, path, transform=None):
        self.imgs_path = path
        self.transform = transform
        self.imgs = glob.glob(f"{self.imgs_path}/*")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = default_loader(self.imgs[index])
        if self.transform is not None:
            image = self.transform(image)
        return image

## uncomment all the following lines to use IETK transformations; make sure all the necessary libraries are installed.

# from ietk.methods import brighten_darken
# from ietk.util import get_foreground
# import numpy as np
# import logging
# from random import random

# class DisableLogger:
#     """
#     Allow to handle log warnings when they are expected 
#     ("ietk" module displays warnings for some .tif images)
#     """
#     def __enter__(self):
#         logging.disable(logging.CRITICAL)
#     def __exit__(self, exit_type, exit_value, exit_traceback):
#         logging.disable(logging.NOTSET)


# class Ietk_transform:
#     """
#     Encapsulate "ietk" module to be used with Pytorch transforms

#     Parameters
#     ----------
#     method : str
#         augmentation method to be used (according to https://github.com/adgaudio/ietk-ret)
#     p : float between 0 and 1
#         probability to skip/apply the augmentation
#     Returns
#     -------
#     img: numpy array
#         Augmented image
#     """
#     def __init__(self, method:str,p=0.5):
#         self.method = method
#         self.p = p
        
#     def __call__(self, image: np.array) -> np.array:
#         img = image/np.max(image)
#         if random() < self.p:
#             try:
#                 with DisableLogger():
#                     fg = get_foreground(img)
#                     img = brighten_darken(img, self.method, focus_region=fg)
#             except:
#                 print("impossible to apply filters on this image")
#         return img



## classes used for segmentation task (work still in progress)

class inference_simple_dt(Dataset):

    def __init__(self, path, feature_extractor):

        self.path = path
        self.feature_extractor = feature_extractor
        self.transforms = A.Resize(height=1200,width=1200)

        self.images_names = os.listdir(self.path)   
        self.images = np.sort(np.array(self.images_names))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = cv2.imread(os.path.join(self.path, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_aug = self.transforms(image=image)["image"]

        encoded_inputs = self.feature_extractor(Image.fromarray(np.uint8(img_aug)),
                                                return_tensors="pt")
        
        for k,_ in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension
        
        return encoded_inputs, self.images[idx]
    

class SegformerForSemanticSegmentation_overriden(SegformerForSemanticSegmentation):
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        criterion = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )





