
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score

# Define constants
NUM_CLASSES = 3  # Rock, Paper, Scissors
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/Users/mom/Desktop/GitHub/rock-paper-scissors/data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_RATIO = 0.7  # 70% train
VAL_RATIO = 0.15   # 15% val, remaining 15% test

# Function to prepare dataset (split into train/val/test if not already done)
def prepare_dataset():
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR) and os.path.exists(TEST_DIR):
        print("Train, val, and test directories already exist. Skipping split.")
        return

    classes = ['rock', 'paper', 'scissors']
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Directory {cls_dir} not found. Ensure data is in 'data/rock', 'data/paper', 'data/scissors'.")

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        total = len(images)
        train_split = int(TRAIN_RATIO * total)
        val_split = int(VAL_RATIO * total) + train_split  # Train + Val

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # Create train, val, and test subdirs
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        val_cls_dir = os.path.join(VAL_DIR, cls)
        test_cls_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Copy files (to preserve originals)
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_cls_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(test_cls_dir, img))

    print("Dataset copied and split into 'data/train', 'data/val', and 'data/test'.")

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Main function
def main():
    # Prepare data (split if needed)
    prepare_dataset()

    # Load datasets
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)  # Use val_transform for test

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class names
    class_names = train_dataset.classes  # ['paper', 'rock', 'scissors'] (order may vary)

    # Load pre-trained EfficientNet-B0
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    # Validation/Test loop (reusable)
    def evaluate(model, loader, criterion, device, dataset_name="Val"):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"{dataset_name} Loss: {running_loss / len(loader):.4f} - {dataset_name} Acc: {acc:.4f}")
        return running_loss / len(loader), acc

    # Main training
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}")
        evaluate(model, val_loader, criterion, DEVICE, "Val")

    # Final evaluation on test set
    print("\nEvaluating on Test Set:")
    evaluate(model, test_loader, criterion, DEVICE, "Test")

    # Save the model
    torch.save(model.state_dict(), '/Users/mom/Desktop/Github/rock-paper-scissors/rps_model.pth')
    print("Model saved as 'rps_model.pth'")

if __name__ == "__main__":
    main()
