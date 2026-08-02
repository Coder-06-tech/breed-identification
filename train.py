# train.py
import os
import argparse
import random
import shutil
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def prepare_split(src_root, out_root="data", val_split=0.2, seed=42):
    """Split dataset into train/val if not already split"""
    random.seed(seed)
    os.makedirs(out_root, exist_ok=True)
    train_root = os.path.join(out_root, "train")
    val_root = os.path.join(out_root, "val")

    for d in (train_root, val_root):
        os.makedirs(d, exist_ok=True)

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    print("Found classes:", classes)

    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        files = [f for f in glob(os.path.join(src_dir, "*")) if os.path.isfile(f)]
        random.shuffle(files)
        split_idx = int(len(files) * (1 - val_split))
        train_files, val_files = files[:split_idx], files[split_idx:]

        os.makedirs(os.path.join(train_root, cls), exist_ok=True)
        os.makedirs(os.path.join(val_root, cls), exist_ok=True)

        for f in train_files:
            shutil.copy2(f, os.path.join(train_root, cls, os.path.basename(f)))
        for f in val_files:
            shutil.copy2(f, os.path.join(val_root, cls, os.path.basename(f)))

    return train_root, val_root

def train(args):
    if os.path.isdir(os.path.join(args.dataset, "train")) and os.path.isdir(os.path.join(args.dataset, "val")):
        train_dir = os.path.join(args.dataset, "train")
        val_dir = os.path.join(args.dataset, "val")
    else:
        print("No train/val split found, creating one...")
        train_dir, val_dir = prepare_split(args.dataset, out_root="data", val_split=args.val_split)

    # Advanced data augmentation for higher accuracy
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    classes = train_ds.classes
    print("Classes:", classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training")

    # Load appropriate pretrained weights and set classifier head
    if args.arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Fine-tuning: Unfreeze layer4 and fully-connected head
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        
    elif args.arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Fine-tuning: Unfreeze features.16 and classifier head
        for name, param in model.named_parameters():
            if "features.16" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        # Step the learning rate scheduler
        scheduler.step()
        
        epoch_acc = running_corrects / len(train_ds)
        epoch_loss = running_loss / len(train_ds)
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        model.eval()
        val_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                val_corrects += torch.sum(preds == labels.data).item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_corrects / len(val_ds)
        print(f"Val Acc: {val_acc:.4f}")

        # Save confusion matrix plot
        cm = confusion_matrix(all_labels, all_preds)
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        plt.close()

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "classes": classes
            }, args.save_path)
            print(f"✅ Saved best model (acc={best_acc:.4f}) to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="models/model.pth")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_large"], help="Neural network architecture")
    args = parser.parse_args()
    train(args)
