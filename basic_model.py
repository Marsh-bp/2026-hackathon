#Run this to get dataset
import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

print("Path to dataset files:", path)

##############################################
# Running Only one batch to verify the "Basic" setup works
# Run below code for preprocessing step it takes 
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification

base_path = '/kaggle/input/cifake-real-and-ai-generated-synthetic-images'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')

# ViT requires 224x224 images
model_name = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name)

# Transformations: Resize 32x32 to 224x224 and normalize
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

# Basic DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


###################################################################################

# Run below code to load pretrained data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ViT and replace the head for 2 classes (FAKE, REAL)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)

#############################################################################

# Run this to know initial training loss

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images).logits
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Initial Training Loss: {loss.item():.4f}")
    break
##############################################################################

# Run this for evalution on test set (whole data is taken)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model.eval()
all_preds = []
all_labels = []

print("Running Final Evaluation on Test Set...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 1. Print Standard Metrics
print("\n--- PHASE 1 EVALUATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=['FAKE', 'REAL']))

# 2. Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Phase 1: Confusion Matrix')
plt.show()

