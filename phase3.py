# ======================================================
# PHASE 3: IMPROVE (Adversarial Training)
# ======================================================

import torch
from tqdm.auto import tqdm
from transformers import ViTForImageClassification

# 1. RE-ESTABLISH ENVIRONMENT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-define the attack function needed for training
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# Ensure the model is in memory
if 'model' not in locals():
    print("Model not found in this cell's memory. Re-loading...")
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True, output_attentions=True
    ).to(device)

print(f"Starting Phase 3: Improve on {device}...")

# 2. SETUP
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
criterion = torch.nn.CrossEntropyLoss()

# 3. IMPROVEMENT LOOP (Adversarial Training)
# We show the model both clean and attacked images so it learns to be robust.


model.train()
for epoch in range(1):
    robust_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Robust Training"):
        images, labels = images.to(device), labels.to(device)
        
        # adversarial noise "on the fly"
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        model.zero_grad()
        loss.backward()
        
        # attacked versions of the images
        data_grad = images.grad.data
        adv_images = fgsm_attack(images, 0.05, data_grad).detach()
        
        # Train on a mix of clean and adversarial data
        combined_images = torch.cat([images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        
        optimizer.zero_grad()
        outputs = model(combined_images)
        loss = criterion(outputs.logits, combined_labels)
        loss.backward()
        optimizer.step()
        robust_loss += loss.item()

    print(f"Phase 3 Training Loss: {robust_loss/len(train_loader):.4f}")

# 4. FINAL RE-EVALUATION
print("\n--- FINAL TEST: PERFORMANCE UNDER ATTACK ---")
model.eval()
for eps in [0.05, 0.1]:
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        model.zero_grad()
        loss.backward()
        
        # Test how the improved model handles the attack
        adv_imgs = fgsm_attack(images, eps, images.grad.data)
        with torch.no_grad():
            outputs = model(adv_imgs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += preds.eq(labels).sum().item()
    
    print(f"Robust Accuracy at Epsilon {eps}: {correct/len(test_dataset):.4f}")
