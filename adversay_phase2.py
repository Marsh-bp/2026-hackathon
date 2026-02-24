# 7. PHASE 2: BREAK (Adversarial Attack)
def fgsm_attack(image, epsilon, data_grad):
    # Collect the sign of the gradients
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel
    perturbed_image = image + epsilon * sign_data_grad
    # Return the perturbed image
    return torch.clamp(perturbed_image, 0, 1)

def break_model(model, device, test_loader, epsilon):
    model.eval()
    adv_examples = []
    correct = 0

    for images, labels in tqdm(test_loader, desc=f"Breaking (Eps={epsilon})"):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True # Required to get gradients for attack

        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        model.zero_grad()
        loss.backward()

        # Get data gradients
        data_grad = images.grad.data

        # Create perturbed images
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Re-classify the perturbed images
        with torch.no_grad():
            outputs = model(perturbed_data)
            final_pred = outputs.logits.max(1, keepdim=True)[1]

            # Check for success
            # Focus on: True Label = FAKE (0), Predicted = REAL (1)
            for i in range(len(labels)):
                if final_pred[i].item() == labels[i].item():
                    correct += 1
                elif labels[i].item() == 0 and final_pred[i].item() == 1:
                    # Successfull
                    if len(adv_examples) < 5:
                        adv_examples.append((images[i], perturbed_data[i], final_pred[i]))

    final_acc = correct / len(test_loader.dataset)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc:.4f}")
    return adv_examples

# Run the attack
epsilons = [0.05, 0.1, 0.2] # Strength of the attack
for eps in epsilons:
    break_model(model, device, test_loader, eps)
