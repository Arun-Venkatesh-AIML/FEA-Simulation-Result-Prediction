import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.autograd as autograd
import matplotlib.pyplot as plt

# Import parsing and model definitions from your existing modules
from stored_functions import (
    parse_fem_file,
    load_simulation_output,
    export_results_to_h5,
    NodesDataset,
    ElasticNet
)


def pinn_loss_fn(model, batch_inputs, batch_targets, alpha_phys=1.0, in_std=None):
    """
    Args:
        model         : Neural network
        batch_inputs  : Normalized input [N, 13]
        batch_targets : Normalized displacement [N, 3]
        alpha_phys    : Weight for physics loss
        in_std        : Tensor of input std [13] from dataset — required to rescale gradients
    """
    preds = model(batch_inputs)
    mse_data = nn.MSELoss()(preds, batch_targets)

    u = preds[:, 0:1]
    v = preds[:, 1:2]
    w = preds[:, 2:3]

    # Compute autograd derivatives w.r.t. normalized inputs
    grads_u = autograd.grad(u, batch_inputs, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    grads_v = autograd.grad(v, batch_inputs, torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    grads_w = autograd.grad(w, batch_inputs, torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    # 🔧 Rescale spatial derivative dimensions [0:x, 1:y, 2:z]
    if in_std is not None:
        # Ensure in_std shape is [1, 13]
        if len(in_std.shape) == 1:
            in_std = in_std.unsqueeze(0)
        grads_u[:, 0:3] *= in_std[0, 0:3]
        grads_v[:, 0:3] *= in_std[0, 0:3]
        grads_w[:, 0:3] *= in_std[0, 0:3]

    du_dx, du_dy, du_dz = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]
    dv_dx, dv_dy, dv_dz = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]
    dw_dx, dw_dy, dw_dz = grads_w[:, 0:1], grads_w[:, 1:2], grads_w[:, 2:3]

    E = batch_inputs[:, 8:9]
    nu = batch_inputs[:, 9:10]
    lam = (E * nu) / ((1 + nu)*(1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    trace_strain = du_dx + dv_dy + dw_dz

    res_x = mu * (2 * du_dx + dv_dy + dw_dz) + lam * trace_strain
    res_y = mu * (du_dy + 2 * dv_dy + dw_dz) + lam * trace_strain
    res_z = mu * (du_dz + dv_dz + 2 * dw_dz) + lam * trace_strain

    physics_loss = (res_x**2 + res_y**2 + res_z**2).mean()

    total_loss = mse_data + alpha_phys * physics_loss
    return total_loss, mse_data.item(), physics_loss.item()


def train_model_with_pinn(all_fem_paths, all_pch_paths, test_size, num_epochs, batch_size, lr, device, alpha_phys=1.0):
    train_fem, test_fem, train_pch, test_pch = train_test_split(all_fem_paths, all_pch_paths, test_size=test_size)
    train_dataset = NodesDataset(train_fem, train_pch)
    test_dataset = NodesDataset(test_fem, test_pch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_mean, in_std = train_dataset.in_mean.to(device), train_dataset.in_std.to(device)
    out_mean, out_std = train_dataset.out_mean.to(device), train_dataset.out_std.to(device)

    model = ElasticNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'total': [], 'data': [], 'phys': [], 'mae': [], 'rmse': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        data_loss_epoch = 0.0
        phys_loss_epoch = 0.0

        for inputs_norm, targets_norm in train_loader:
            inputs_norm = inputs_norm.to(device).requires_grad_(True)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            loss, l_data, l_phys = pinn_loss_fn(model, inputs_norm, targets_norm, alpha_phys, in_std)
            loss.backward()
            optimizer.step()

            bsize = inputs_norm.size(0)
            total_loss_epoch += loss.item() * bsize
            data_loss_epoch  += l_data * bsize
            phys_loss_epoch  += l_phys * bsize

        n_train = len(train_loader.dataset)
        history['total'].append(total_loss_epoch / n_train)
        history['data'].append(data_loss_epoch / n_train)
        history['phys'].append(phys_loss_epoch / n_train)

        # --- Evaluation on test set ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y_true.append(y.cpu().numpy())
                y_pred.append(model(x).cpu().numpy())

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        history['mae'].append(mae)
        history['rmse'].append(rmse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Total={history['total'][-1]:.4e}, MSE={history['data'][-1]:.2e}, Phys={history['phys'][-1]:.2e}, MAE={mae:.2e}, RMSE={rmse:.2e}")

    print("✅ PINN training complete.\n")
    plot_training_history(history)

    # ---- Save artifacts ----
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "displacement_model_with_pinn.pth")
    stats_path = os.path.join("models", "stats.pkl")

    torch.save(model.state_dict(), model_path)
    stats = {
        'in_mean': in_mean.cpu().numpy(),
        'in_std': in_std.cpu().numpy(),
        'out_mean': out_mean.cpu().numpy(),
        'out_std': out_std.cpu().numpy()
    }
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print(f"📦 Model saved to: {model_path}")
    print(f"📊 Normalization stats saved to: {stats_path}")

    return model, in_mean, in_std, out_mean, out_std

def plot_training_history(history):
    # Plot data and physics losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['data'], label='Data Loss (MSE)', linewidth=2)
    plt.plot(history['phys'], label='Physics Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.title("Training Loss: Data vs Physics")
    plt.tight_layout()
    plt.show()

    # Plot test performance metrics
    plt.figure(figsize=(10, 4))
    plt.plot(history['mae'], label='Test MAE', linewidth=2)
    plt.plot(history['rmse'], label='Test RMSE', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title("Test Set Evaluation Metrics")
    plt.tight_layout()
    plt.show()

    # --- Main ---
if __name__ == "__main__":
    print("▶ Starting data-driven training pipeline")

    base_data_dir = r"O:\AIML\SEMESTER_4\DEPLOYMENT_PINN_MODEL\data"
    fem_pattern = os.path.join(base_data_dir, "fem", "*.fem")
    pch_pattern = os.path.join(base_data_dir, "pch", "*.pch")

    all_fem_paths = sorted(glob.glob(fem_pattern))
    all_pch_paths = sorted(glob.glob(pch_pattern))
    assert len(all_fem_paths) == len(all_pch_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, in_mean, in_std, out_mean, out_std = train_model_with_pinn(
        all_fem_paths=all_fem_paths,
        all_pch_paths=all_pch_paths,
        test_size=0.2,
        num_epochs=500,
        batch_size=1024,
        lr=1e-3,
        device=device,
        alpha_phys=0,
    )

