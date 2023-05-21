import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from utils import DataAugmentation, Efficient_b2, EnsembleModel, Resnet18
from torchvision import transforms


def testing(model, test_loader, device):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for k, test_data in enumerate(tqdm(test_loader, desc="testing batches")):
            inputs, labels = test_data
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = inputs.type(torch.float)
            output = model(inputs)

            labels = labels.unsqueeze(1).type(torch.float)
            y_pred.extend(output.tolist())

    return np.array(y_pred).ravel()

def plot_AUC(y_true,y_pred, model_name):
    auc = roc_auc_score(y_true,y_pred)
    ran = [0 for _ in range(len(y_true))]
    ran_fpr, ran_tpr, _ = roc_curve(y_true, ran)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)

    print("AUC = ", auc)
    plt.plot(ran_fpr, ran_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    plt.savefig('results/AUC.png')
    plt.clf()

def main(args):

    device = torch.device(args.device)
    model_name = args.model
    model = globals()[model_name]()
    if model_name == "EnsembleModel":
        print(f"loading pretrained weights from: results/pretrained_EnsembleModel.pt")
        model.load_state_dict(torch.load(f"results/pretrained_EnsembleModel.pt"))
    model.to(device)
    y_preds = []
    for size in args.image_size:
        transform_test = transforms.Compose(
                    [
                        transforms.Resize((size,size)),
                        transforms.ToTensor(),
                    ]
                )
        
        test_set = ImageFolder(args.path_to_dataset, transform=transform_test)

        test_loader = DataLoader(
                            test_set, 
                            batch_size=1,
                            drop_last=False
                        )
    
        classes = test_set.classes
        print(classes)
        _, counts = np.unique(test_set.targets, return_counts=True)
        print("number of samples per class: ", counts)

        y_pred = testing(model, test_loader, device)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    print(y_preds.shape)
    y_pred = np.mean(y_preds, axis=0)
    print(y_pred.shape)

    y_true = np.array(test_set.targets)
    plot_AUC(y_true,y_pred, model_name)

    y_bin = np.where(y_pred > 0.5, 1, 0)

    print("kappa score: ", cohen_kappa_score(y_true,y_bin))
    print("f1 score: ", f1_score(y_true,y_bin))
    print("precision: ", precision_score(y_true,y_bin))
    print("recall: ", recall_score(y_true,y_bin))
    print("accuracy: ", accuracy_score(y_true,y_bin))

    plt.rcParams['figure.figsize'] = [15, 11]
    cf_matrix = confusion_matrix(y_true, y_bin, normalize="true")
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * len(classes), index = [i for i in classes],
                        columns = [i for i in classes])
    sn.set(font_scale=1.4)
    sns_plot = sn.heatmap(df_cm, annot=True)
    plt.savefig('results/heatmap.png')
    img_names = [i[0].split('/')[-1] for i in test_set.imgs]

    df_res = pd.DataFrame({'images':img_names,'groundtruth': y_true.ravel(), f'predictions': y_bin.ravel(), f'softmax_output': y_pred.ravel()})

    df_res.to_csv(f'results/predictions_{model_name}.csv')
    print(f"created csv with results in results/predictions_{model_name}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--path-to-dataset", type=str, default="data")
    parser.add_argument("-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("-m", "--model", type=str, default='Resnet18')
    parser.add_argument("-s", "--image-size", type=list, default=[420])
    parser.add_argument("-l", "--autres", type=int, default=420)
    args = parser.parse_args()
    print(vars(args))
    print("########################")
    main(args)