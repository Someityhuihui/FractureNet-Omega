"""
Physics-Informed Parameter Identifier
=======================================
CNN encoder that extracts material & crack parameters from displacement fields.

Input:  displacement field (B, 2, H, W) + optional phase-field (B, 1, H, W)
Output: 8 parameters [E, nu, Gf, ft, a, theta, K_I, P]

Physics constraints enforce:
  - K = E/4 * sqrt(pi/(2a)) * COD  (LEFM consistency)
  - G = K^2/E <= G_f               (Griffith energy bound)
  - 0 < nu < 0.5                   (Poisson ratio bounds)
"""

import torch
import torch.nn as nn

PARAM_NAMES = [
    'elastic_modulus',       # 0: E (GPa)
    'poisson_ratio',         # 1: nu (-)
    'fracture_energy',       # 2: Gf (N/mm)
    'tensile_strength',      # 3: ft (MPa)
    'crack_length',          # 4: a (mm)
    'crack_angle',           # 5: theta (deg)
    'stress_intensity_K',    # 6: KI (MPa*sqrt(mm))
    'load_magnitude',        # 7: P (kN)
]

PARAM_RANGES = {
    'elastic_modulus':    (10, 200),
    'poisson_ratio':      (0.1, 0.45),
    'fracture_energy':    (0.01, 2.0),
    'tensile_strength':   (0.5, 10.0),
    'crack_length':       (1, 50),
    'crack_angle':        (-90, 90),
    'stress_intensity_K': (0, 100),
    'load_magnitude':     (0, 500),
}


class PhysicsInformedEncoder(nn.Module):
    """CNN encoder: displacement field → latent features."""

    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, latent_dim, 4, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)


class ParameterRegressor(nn.Module):
    """MLP head: latent → 8 physical parameters."""

    def __init__(self, latent_dim=512, hidden=256, n_params=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_params)
        )

    def forward(self, z):
        return self.net(z)


class PhysicsInformedParameterIdentifier(nn.Module):
    """
    Full identifier: encoder + parameter head + physics head.
    """

    def __init__(self, in_channels=3, latent_dim=512, n_params=8):
        super().__init__()
        self.encoder = PhysicsInformedEncoder(in_channels, latent_dim)
        self.param_head = ParameterRegressor(latent_dim, 256, n_params)
        self.physics_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        params = self.param_head(z)
        phys_residual = self.physics_head(z)
        return params, phys_residual


class PhysicsLoss(nn.Module):
    """
    Composite loss: MSE + LEFM consistency + energy bound + param bounds.
    """

    def __init__(self, lambda_lefm=0.1, lambda_energy=0.05, lambda_bounds=0.01):
        super().__init__()
        self.lambda_lefm = lambda_lefm
        self.lambda_energy = lambda_energy
        self.lambda_bounds = lambda_bounds

    def forward(self, u_pred, params_pred, u_true, params_true=None):
        losses = {}

        # Data fidelity
        losses['data'] = nn.functional.mse_loss(u_pred, u_true)

        # Supervised parameter loss (if labels available)
        if params_true is not None:
            losses['supervised'] = nn.functional.mse_loss(params_pred, params_true)

        # LEFM consistency: K = E/4 * sqrt(pi/(2*a+eps)) * COD
        E, a, K = params_pred[:, 0], params_pred[:, 4], params_pred[:, 6]
        cod = self._compute_cod(u_pred)
        K_pred = E / 4.0 * torch.sqrt(torch.pi / (2.0 * a + 1e-8)) * cod
        losses['lefm'] = nn.functional.mse_loss(K_pred, K)

        # Griffith bound: G = K^2/E <= Gf
        G_plane = K**2 / (E + 1e-8)
        Gf = params_pred[:, 2]
        losses['energy'] = torch.mean(torch.relu(G_plane - Gf))

        # Parameter range penalty
        losses['bounds'] = 0.0
        for i, (lo, hi) in enumerate([
            (10, 200), (0.1, 0.45), (0.01, 2.0), (0.5, 10.0),
            (1, 50), (-90, 90), (0, 100), (0, 500)
        ]):
            losses['bounds'] += torch.mean(torch.relu(lo - params_pred[:, i]))
            losses['bounds'] += torch.mean(torch.relu(params_pred[:, i] - hi))

        # Weighted total
        total = losses['data']
        total += self.lambda_lefm * losses['lefm']
        total += self.lambda_energy * losses['energy']
        total += self.lambda_bounds * losses['bounds']
        if params_true is not None:
            total += losses['supervised']

        losses['total'] = total
        return losses

    def _compute_cod(self, u):
        """Crack opening displacement from displacement field."""
        center = u.shape[-1] // 2
        return torch.mean(torch.abs(u[:, 0, :, center] - u[:, 1, :, center]), dim=1)


# ================================================================
# Training
# ================================================================
def train_identifier(model, dataloader, epochs=100, lr=1e-4, device='cpu'):
    """Train the parameter identifier."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PhysicsLoss()
    history = []

    for epoch in range(epochs):
        epoch_losses = []
        for batch in dataloader:
            u_true = batch['displacements'].to(device)
            params_true = batch.get('params')
            if params_true is not None:
                params_true = params_true.to(device)

            params_pred, _ = model(u_true)
            # Simple reconstruction: u_pred = u_true for now
            # In production: use learned forward model
            losses = loss_fn(u_true, params_pred, u_true, params_true)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            epoch_losses.append({k: v.item() for k, v in losses.items()})

        history.append(epoch_losses)
        if epoch % 20 == 0:
            avg = {k: sum(h[k] for h in epoch_losses) / len(epoch_losses)
                   for k in epoch_losses[0]}
            print(f"Epoch {epoch:3d}: total={avg['total']:.4f}, "
                  f"data={avg['data']:.4f}, lefm={avg['lefm']:.4f}")

    return model, history


# ================================================================
# Inference
# ================================================================
def identify_parameters(model, displacement_field, device='cpu'):
    """
    Given a displacement field, identify material and crack parameters.

    Args:
        model: trained PhysicsInformedParameterIdentifier
        displacement_field: (1, 2, H, W) or (2, H, W) tensor

    Returns:
        dict with parameter names and values
    """
    model.eval()
    if displacement_field.dim() == 3:
        displacement_field = displacement_field.unsqueeze(0)

    with torch.no_grad():
        params, phys_res = model(displacement_field.to(device))

    result = {}
    for i, name in enumerate(PARAM_NAMES):
        result[name] = float(params[0, i].cpu())

    result['physics_violation'] = float(phys_res[0, 0].cpu())
    return result


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  PhysicsInformedParameterIdentifier — Test")
    print("=" * 50)

    # Dummy test
    model = PhysicsInformedParameterIdentifier(in_channels=3)
    x = torch.randn(4, 3, 128, 128)
    params, phys = model(x)
    print(f"Input:  {x.shape}")
    print(f"Params: {params.shape} — {PARAM_NAMES[:4]}...")
    print(f"Physics residual: {phys.shape}")

    # Loss test
    loss_fn = PhysicsLoss()
    losses = loss_fn(x, params, x)
    print(f"Losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")

    # Param ranges check
    print(f"\nParameter ranges:")
    for i, name in enumerate(PARAM_NAMES):
        vals = params[:, i].detach()
        print(f"  {name:25s}: [{vals.min():.2f}, {vals.max():.2f}]  "
              f"(allowed: {PARAM_RANGES[name]})")

    print("\nIdentifier module ready!")
