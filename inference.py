import argparse


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm
from transformers import SegformerFeatureExtractor

import utils
from utils import EnsembleModel


def segmentation(model,loader):
    with torch.no_grad():
        for j, batch in enumerate(tqdm(loader, desc="inference ...")):
            data , image_name = batch
            image = data["pixel_values"]

            outputs = model(pixel_values=image)
            logits = outputs.logits

            upsampled_logits = nn.functional.interpolate(logits, size=(512,512), mode="bilinear", align_corners=False)
            p = torch.sigmoid(upsampled_logits)
            p = p[0][0].numpy()*255
            cv2.imwrite(f"pred_test/{image_name[0].split('.')[0]+'.png'}", p)


def testing(model, inference_loader, device):
    y_pred = []

    model.eval()

    with torch.no_grad():
        for k, test_data in enumerate(tqdm(inference_loader, desc="testing batches")):
            inputs = test_data
            inputs = inputs.to(device)
            inputs = inputs.type(torch.float)
            output = model(inputs)
            y_pred.extend(output.tolist())

    return np.array(y_pred)


def get_pred(y_pred1, y_pred2, i):
    if y_pred1[i] < 0.5:
        return 0, y_pred1[i]
    if y_pred2[i] < 0.5:
        return 1, y_pred2[i]
    return 2, y_pred2[i]

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EnsembleModel()
    model.to(device)

    model.load_state_dict(torch.load(f"results/pretrained_EnsembleModel1.pt"))
    print('loaded EnsembleModel weights from checkpoint for classification task1')
    y_preds = []
    for size in args.image_size:
        print("resizing to: ", size)
        transform_inference = transforms.Compose(
                    [
                        transforms.Resize((size,size)),
                        transforms.ToTensor(),
                    ]
                )
        
        inference_set = utils.InferenceBasic(args.images_path, transform=transform_inference)

        inference_loader = DataLoader(
                            inference_set, 
                            batch_size=1,
                            drop_last=False
                        )
        y_pred = testing(model, inference_loader, device)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    y_pred1 = np.mean(y_preds, axis=0)
    y_pred1 = 1 - y_pred1
    y_bin1 = np.where(y_pred1 > 0.5, 1, 0)

    model.load_state_dict(torch.load(f"results/pretrained_EnsembleModel2.pt"))
    print('loaded EnsembleModel weights from checkpoint for classification task2')
    y_pred2 = testing(model, inference_loader, device)
    y_pred2 = 1 - y_pred2
    y_bin2 = np.where(y_pred2 > 0.5, 1, 0)

    torch.cuda.empty_cache()

    predictions = np.ones((len(inference_set),4))
    conv = ['small/none', 'intermediate','large']

    for i in range(len(inference_set)):
        prediction, confidence = get_pred(y_pred1, y_pred2, i)
        confidence = int(abs(0.5 - confidence)*100/0.5)
        predictions[i][0] = prediction
        predictions[i][1] = confidence
    
    images = [i.split('/')[-1] for i in inference_set.imgs]
    df_res = pd.DataFrame({'images':images,
                        'predictions1': y_bin1.ravel(), 
                        'softmax_output1': y_pred1.ravel(), 
                        'predictions2': y_bin2.ravel(), 
                        'softmax_output2': y_pred2.ravel(), 
                        'predictions_final': predictions[:,0], 'confidence':predictions[:,1]})
    
    df_res.to_csv(f'results/inferences.csv')
    print(f"created csv with results in results/inferences.csv")


    ###### segmentation part #######
    visu = input("Do you want to visualize the images and their predictions / segmentation masks? (y/n)")
    
    if "y" in visu:

        feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)

        dataset = utils.inference_simple_dt(path=args.images_path, 
                                        feature_extractor=feature_extractor)
        
        print("Number of samples:", len(dataset))
        
        loader = DataLoader(dataset, 
                            batch_size=1,
                            drop_last=False
                            )
        
        id2label = {1:'drusen'}
        label2id = {v: k for k, v in id2label.items()}

        model = utils.SegformerForSemanticSegmentation_overriden.from_pretrained(f"results/weights_weivb1",
                                                                                    id2label=id2label,
                                                                                    label2id=label2id,
                                                                                )  
        print('loaded segformer weights from checkpoint for segmentation task')
        segmentation(model,loader)


        for i in range (len(inference_set)):
            follow = input('Press "enter" to see next image or "e" to exit program')
            if "e" in follow:
                break
            name = images[i]
            img_d = cv2.cvtColor(cv2.imread(inference_set.imgs[i]), cv2.COLOR_BGR2RGB)
            img_d = cv2.resize(img_d, (512,512))
            seg_mask = cv2.imread(f'pred_test/{name.split(".")[0]}.png', cv2.IMREAD_GRAYSCALE)

            fig = plt.figure()
            fig.suptitle(f'FILENAME:{name}, PREDICTION: {predictions[i][0]} ({conv[int(predictions[i][0])]}), CONFIDENCE: {predictions[i][1]}% [{y_pred1[i]},{y_pred2[i]}]', fontsize=16)

            ax1 = plt.subplot(1,2,1)
            ax1.set_title(f"Image {name}")
            ax1.imshow(img_d, interpolation='none')

            ax2 = plt.subplot(1,2,2)
            ax2.set_title(f"Image {name} with drusen segmentation mask")
            ax2.imshow(img_d, interpolation='none')
            if predictions[i][0] > 0:
                ax2.imshow(seg_mask, interpolation='none', alpha=0.25)
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "inferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--images-path", type=str, default="test_complet")
    parser.add_argument("-s", "--image-size", type=list, default=[350,420,512])
    args = parser.parse_args()
    print(vars(args))
    print("########################")
    main(args)
