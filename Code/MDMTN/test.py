from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I
import torch
import matplotlib.pyplot as plt
from train_and_test_model_MM import load_MultiMnist_data

if __name__ == '__main__':
    # Data
    train_loader, val_loader, test_loader = load_MultiMnist_data()
    images, targets = next(iter(test_loader))

    # Model
    mod_params_mgda = {"batch_size": 100}
    device = "cpu"
    model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
    model.load_model("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/11-04-2025--18-14-52/model_0")
    model.eval()

    # Predict
    images = images.unsqueeze(-1)
    outputs = model(images)
    task1_outputs = outputs[:, :10]
    task2_outputs = outputs[:, 10:]

    plt.figure(figsize=(15, 6))
    for i in range(10):
        # Get predictions
        pred1 = task1_outputs[i].argmax().item()
        pred2 = task2_outputs[i].argmax().item()

        # Get ground truth
        true_left = targets[0][i].item()
        true_right = targets[1][i].item()

        # Plot
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(f'Label: {true_left} {true_right} | Pred: {pred1} {pred2}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()