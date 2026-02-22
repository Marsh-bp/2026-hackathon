import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
import kagglehub
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

model_name = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Load and Subset Data (Balanced 1000 Train / 200 Test)
def get_balanced_subset_indices(dataset, num_per_class=500):
    indices = []
    targets = np.array(dataset.targets)
    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        subset = np.random.choice(class_indices, num_per_class, replace=False)
        indices.extend(subset)
    return indices

full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
full_test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

train_indices = get_balanced_subset_indices(full_train_dataset, num_per_class=500)
test_indices = get_balanced_subset_indices(full_test_dataset, num_per_class=100)

train_dataset = Subset(full_train_dataset, train_indices)
test_dataset = Subset(full_test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. PHASE 1: BUILD (Model Training)
# ==========================================
print("\n--- PHASE 1: BUILDING DETECTOR ---")
model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True, output_attentions=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(1):
    for images, labels in tqdm(train_loader, desc="Training Phase 1"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

# Initial Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nPHASE 1 REPORT:")
print(classification_report(all_labels, all_preds, target_names=['FAKE', 'REAL']))

# ==========================================
# 3. PHASE 2: BREAK (Adversarial Attack)
# ==========================================
print("\n--- PHASE 2: BREAKING MODEL ---")

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def evaluate_attack(model, device, loader, epsilon):
    model.eval()
    correct = 0
    for images, labels in tqdm(loader, desc=f"Attacking (Eps={epsilon})"):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        model.zero_grad()
        loss.backward()
        
        perturbed_data = fgsm_attack(images, epsilon, images.grad.data)
        with torch.no_grad():
            outputs = model(perturbed_data)
            final_pred = outputs.logits.max(1, keepdim=True)[1]
            correct += final_pred.eq(labels.view_as(final_pred)).sum().item()
            
    acc = correct / len(loader.dataset)
    print(f"Epsilon: {epsilon} | Test Accuracy: {acc:.4f}")
    return acc

# Test vulnerability
epsilons = [0.05, 0.1]
for eps in epsilons:
    evaluate_attack(model, device, test_loader, eps)

# ==========================================
# 4. PHASE 3: IMPROVE (Adversarial Training)
# ==========================================
print("\n--- PHASE 3: IMPROVING (ROBUST TRAINING) ---")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(1):
    for images, labels in tqdm(train_loader, desc="Robust Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Create adversarial batch
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        model.zero_grad()
        loss.backward()
        adv_images = fgsm_attack(images, 0.05, images.grad.data).detach()
        
        # Train on both
        combined_imgs = torch.cat([images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        
        optimizer.zero_grad()
        outputs = model(combined_imgs)
        loss = criterion(outputs.logits, combined_labels)
        loss.backward()
        optimizer.step()

# ==========================================
# 5. FINAL RE-EVALUATION
# ==========================================
print("\n--- FINAL RESULTS AFTER IMPROVEMENT ---")
for eps in [0.05, 0.1]:
    evaluate_attack(model, device, test_loader, eps)

# Visualization Function
def plot_attention_map(model, img_tensor, label):
    model.eval()
    img_input = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_input)
    attentions = outputs.attentions[-1]
    avg_attn = torch.mean(attentions[0, :, 0, 1:], dim=0).reshape(14, 14).cpu().numpy()
    
    img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = (img_display * np.array(processor.image_mean)) + np.array(processor.image_mean)
    img_display = np.clip(img_display, 0, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_display); plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(avg_attn, cmap='magma'); plt.title("Robust Attention"); plt.axis('off')
    plt.show()

plot_attention_map(model, test_dataset[0][0], test_dataset[0][1])
