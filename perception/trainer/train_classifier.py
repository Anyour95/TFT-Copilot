import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def train(data_dir, epochs=20, batch=32, lr=1e-3, out='classifier.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tf_val)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4)

    num_classes = len(train_ds.classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        epoch_loss = running / len(train_ds)

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outs = model(imgs)
                _, preds = torch.max(outs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f'Epoch {epoch}/{epochs} loss={epoch_loss:.4f} val_acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out)
            print(f'Saved best model to {out} (acc={best_acc:.4f})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='classifier_data', help='classifier data root (train/ val folders)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='classifier.pth')
    args = p.parse_args()

    train(args.data, epochs=args.epochs, batch=args.batch, lr=args.lr, out=args.out)

if __name__ == '__main__':
    main()
