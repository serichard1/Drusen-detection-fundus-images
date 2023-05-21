import argparse
from operator import itemgetter
from statistics import mean

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils import DataAugmentation, Efficient_b2, EnsembleModel, Resnet18


def train(model,train_loader,optimizer,criterion, scaler, device):
    m = {"loss":[],"accuracy":[]}
    model.train()
    for _, training_data in enumerate(tqdm(train_loader, desc="training batches: ")):
        inputs, labels = training_data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.type(torch.float)

        optimizer.zero_grad()
        
        outputs = model(inputs)

        labels = labels.unsqueeze(1).type(torch.float)

        loss = criterion(outputs, labels)
        scaler.scale(loss).backward()

        m["loss"].append(loss.item())

        scaler.step(optimizer)
        scaler.update()
        predicted = torch.round(outputs.data)
        m["accuracy"].append(accuracy_score(torch.Tensor.cpu(labels),torch.Tensor.cpu(predicted)))
    
    m["loss"] = mean(m["loss"])
    m["accuracy"] = mean(m["accuracy"])
    return m


def validation(model,valid_loader,criterion, device):
    m = {"loss":[],"accuracy":[]}
    model.eval()
    with torch.no_grad():
        for _, valid_data in enumerate(tqdm(valid_loader, desc="validation batches")):
            inputs, labels = valid_data

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.float)
            
            outputs = model(inputs)
            labels = labels
            labels = labels.unsqueeze(1).type(torch.float)

            loss = criterion(outputs, labels)
            m["loss"].append(loss.item())
            predicted = torch.round(outputs.data)
            m["accuracy"].append(accuracy_score(torch.Tensor.cpu(labels),torch.Tensor.cpu(predicted)))
        
        m["loss"] = mean(m["loss"])
        m["accuracy"] = mean(m["accuracy"])
    return m

def main(args):
    device = torch.device(args.device)
    model_name = args.model
    model = globals()[model_name](args.pretrained)
    model.to(device)

    transform_train = DataAugmentation(resize=args.image_size, augmentation_type=args.augmentation_type)

    if args.augmentation_type == "albumentations":
        def numpy_loader(path: str):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
        train_set = ImageFolder(args.path_to_dataset, transform=transform_train, loader=numpy_loader)
    else:
        train_set = ImageFolder(args.path_to_dataset, transform=transform_train)

    classes = train_set.classes
    print("classes: ", classes)

    train_size = int(0.77* len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    print("number of samples: ", train_size + valid_size)
    print("train: ", train_size)
    print("valid: ", valid_size)
    
    train_loader = DataLoader(
                        train_set, 
                        batch_size=args.batch_size,
                        drop_last=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        shuffle=True
                    )
    
    valid_loader = DataLoader(
                        valid_set, 
                        batch_size=args.batch_size_eval,
                        drop_last=False,
                        num_workers=args.num_workers
                    )

    _, counts = np.unique(train_set.dataset.targets, return_counts=True)
    print("number of samples per class: ", counts)

    # possibility to use WeightedRandomSampler for imbalanced datasets
    if (counts[0]/counts[1] < 0.5) or (counts[0]/counts[1] > 2):
        print("switching to WeightedRandomSampler (imbalanced classes): ")

        # attribue un poids (inverse du nombre d'occurences) donc + nombreux + weight diminue, varie entre 0 et 1
        weights = 1. / torch.tensor(counts, dtype=torch.float)
        weights_train = list(itemgetter(*train_set.indices)(train_set.dataset.targets))
        samples_weights = weights[weights_train]

        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        print("using weighted sampler for imbalanced classes")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)

    writer = SummaryWriter(args.tensorboard_dir)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print("(batch, channels, height, width): ",images.shape)
    print("labels first batch: ", labels)

    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    lr = 0.0005 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    min_loss = 100
    trigger_times = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.n_epochs):
        training = train(model,train_loader,optimizer,criterion, scaler, device)
        validating = validation(model,valid_loader,criterion, device)
        
        if validating["loss"] < min_loss:
            min_loss = validating["loss"]
            trigger_times = 0
            torch.save(model.state_dict(), f"results/weights_{model_name}.pt") 
        else:
            trigger_times += 1
    
        writer.add_scalars(model_name, {'Loss/train': training["loss"],
                            'Loss/validation': validating["loss"]}, epoch)
        writer.add_scalars(model_name, {'Acc/train': training["accuracy"],
                            'Acc/validation': validating["accuracy"]}, epoch)
        
        print(f'\nEpoch {epoch}')
        print(f'Train Acc. => {round(100 * training["accuracy"],3)}%', end=' | ')
        print(f'Train Loss => {round(training["loss"],5)}')
        print(f'valid Acc.  => {round(100 * validating["accuracy"],3)}%', end=' | ')
        print(f'valid Loss  => {round(validating["loss"],5)} (earlystop => {trigger_times}/{args.patience})')
            
        if trigger_times >= args.patience:
            print('Early stop !')
            break

    print(f'end of training')
    torch.save(model.state_dict(), f"results/weightsE_{model_name}.pt") 

    print(f'best weights have been saved in results/weights_{model_name}')
    print(f'loss and accuracy score curves have been saved in Tensorboard logs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Drusen detection M2 LaBRI/CHU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--path-to-dataset", type=str, default="data")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-s", "--image-size", type=int, default=420)
    parser.add_argument("-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-a","--augmentation-type", type=str, default="standard")
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)
    parser.add_argument("-m", "--model", type=str, default='Resnet18')
    parser.add_argument("--batch-size-eval", type=int, default=16)
    parser.add_argument("--pretrained", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)

    args = parser.parse_args()
    print("arguments: ",vars(args))
    print("########################")
    main(args)