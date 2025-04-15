import torch
from config import get_params_mgda
from src.utils.MGDA_utils import load_MultiMnist_mgda
import matplotlib.pyplot as plt

# ----- Thông số giống khi train -----
model_dir_path = "logs/MDMTN_MM_logs/MGDA_model_logs/model_states"
archi_name = "MDMTN"

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Load tham số và model -----
model, Multimnist_params_mgda = get_params_mgda(archi_name.lower(), model_dir_path, device)

# Load trọng số từ các file riêng biệt
model.load_model(model_dir_path + "/11-04-2025--18-14-52/model_0")
print("Model loaded!")

model.to(device)
model.eval()

# ----- Load dữ liệu inference -----
X_train, X_test, y_train, y_test = load_MultiMnist_mgda()

# ----- Lấy 1 batch và chạy inference -----
batch_size = Multimnist_params_mgda["batch_size"]
img_shp = Multimnist_params_mgda["img_shp"]
X_batch = torch.tensor(X_test[:batch_size], dtype=torch.float32).to(device)
y_batch = torch.tensor(y_test[:batch_size], dtype=torch.float32).to(device)
X_batch = X_batch.reshape([batch_size, 1, img_shp[0], img_shp[1], img_shp[2]])

with torch.no_grad():
    outputs = model(X_batch) 

output1 = torch.argmax(outputs[:, :10], dim=1).cpu()
output2 = torch.argmax(outputs[:, 10:], dim=1).cpu()
y_batch = y_batch.cpu()
gt1 = torch.argmax(y_batch[:, 0], dim=1)
gt2 = torch.argmax(y_batch[:, 1], dim=1)

# ----- Hiển thị ảnh -----
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(X_batch):
        break

    img = X_batch[i].squeeze().cpu().numpy()
    ax.imshow(img, cmap='gray')
    ax.axis("off")

    gt_label = f"{gt1[i].item()} {gt2[i].item()}"
    pred_label = f"{output1[i].item()} {output2[i].item()}"
    ax.set_title(f"GT: {gt_label} | Pred: {pred_label}")

plt.tight_layout()
plt.show()