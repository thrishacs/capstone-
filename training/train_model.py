import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import os

# ✅ Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Advanced Data Augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels (for ResNet)
    transforms.Resize((224, 224)),                # Resize to ResNet input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, shear=5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Load FER-2013 Dataset
train_dataset = torchvision.datasets.ImageFolder(root="../data/train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="../data/test", transform=transform)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ Use Pretrained ResNet-50
model = models.resnet50(pretrained=True)

# ✅ Freeze all layers except last block
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

# ✅ Modify FC Layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 7)  # 7 emotion classes
)

model = model.to(device)

# ✅ Loss, Optimizer, LR Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ✅ MixUp Augmentation Function
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ✅ Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()}")

# ✅ Save Model
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/emotion_model_resnet50.pth")
print("✅ Model saved to ../models/emotion_model_resnet50.pth")

# ✅ Evaluate on Test Set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
