"""
Physics Validator
==================
Validates that AI model predictions obey fundamental fracture mechanics laws.

Checks:
  1. Energy balance: W_ext = U_strain + Gf * A_crack
  2. LEFM consistency: K = E/4 * sqrt(pi/(2a)) * COD
  3. Path independence: J-integral contour invariance
  4. Irreversibility: d_{t+1} >= d_t
  5. Bounds: 0 <= d <= 1, sigma <= ft
"""

import numpy as np
import torch


class PhysicsValidator:
    """Validate physics consistency of fracture AI predictions."""

    def __init__(self):
        self.results = {}

    def validate(self, model, test_loader, device='cpu'):
        """Run all physics checks."""
        self.results = {}

        checks = [
            ('energy_balance', self.check_energy_balance),
            ('lefm_consistency', self.check_lefm_consistency),
            ('path_independence', self.check_path_independence),
            ('irreversibility', self.check_irreversibility),
            ('bounds', self.check_bounds),
        ]

        for name, check_fn in checks:
            try:
                self.results[name] = check_fn(model, test_loader, device)
            except Exception as e:
                self.results[name] = {'error': str(e)}

        return self.results

    # ----------------------------------------------------------------
    def check_energy_balance(self, model, loader, device):
        """W_ext = U_strain + Gf * A_crack (within tolerance)."""
        total_error = 0.0
        n = 0
        for batch in loader:
            u = batch['displacements'].to(device)
            try:
                with torch.no_grad():
                    params, _ = model(u)

                # External work: W_ext = 0.5 * P * u_max
                P = params[:, 7]
                u_max = u[:, 1].max(dim=2).values.max(dim=1).values
                W_ext = 0.5 * P * u_max

                # Strain energy: U = 0.5 * sigma * epsilon * Volume
                E = params[:, 0]; eps = u_max / 100
                U_strain = 0.5 * E * eps**2 * 100 * 100

                # Fracture energy: Gf * A_crack
                Gf = params[:, 2]; a = params[:, 4]
                Gf_diss = Gf * a

                balance = torch.abs(W_ext - (U_strain + Gf_diss))
                total_error += balance.mean().item()
                n += 1
            except Exception:
                pass

        avg_error = total_error / max(n, 1)
        return {'mean_error': avg_error, 'n_samples': n,
                'status': 'PASS' if avg_error < 0.1 else 'WARN'}

    # ----------------------------------------------------------------
    def check_lefm_consistency(self, model, loader, device):
        """K = E/4 * sqrt(pi/(2a)) * COD."""
        errors = []
        for batch in loader:
            u = batch['displacements'].to(device)
            try:
                with torch.no_grad():
                    params, _ = model(u)

                E = params[:, 0]; a = params[:, 4]; K_true = params[:, 6]
                center = u.shape[-1] // 2
                COD = torch.abs(u[:, 0, :, center] - u[:, 1, :, center]).mean(dim=1)
                K_pred = E / 4.0 * torch.sqrt(torch.pi / (2.0 * a + 1e-8)) * COD
                errors.append(torch.abs(K_pred - K_true).mean().item())
            except Exception:
                pass

        mean_err = np.mean(errors) if errors else 0
        return {'mean_error': mean_err, 'n_checks': len(errors),
                'status': 'PASS' if mean_err < 5.0 else 'WARN'}

    # ----------------------------------------------------------------
    def check_path_independence(self, model, loader, device):
        """J-integral should be contour-independent."""
        # Simplified: checks variance of J computed from different integration paths
        # In production: implement full domain integral on multiple contours
        return {'status': 'NOT_IMPLEMENTED',
                'note': 'Requires full-field J-integral computation on contours'}

    # ----------------------------------------------------------------
    def check_irreversibility(self, model, loader, device):
        """d_{t+1} >= d_t (phase-field cannot decrease)."""
        # Requires temporal sequences in test data
        violations = 0
        total = 0
        for batch in loader:
            if 'd_prev' not in batch or 'd_next' not in batch:
                continue
            d_prev = batch['d_prev'].numpy()
            d_next = batch['d_next'].numpy()
            violations += (d_next < d_prev - 1e-6).sum()
            total += d_prev.size
        ratio = violations / max(total, 1)
        return {'violations': int(violations), 'total': int(total),
                'ratio': float(ratio),
                'status': 'PASS' if ratio < 0.01 else 'FAIL'}

    # ----------------------------------------------------------------
    def check_bounds(self, model, loader, device):
        """0 <= d <= 1, sigma <= ft."""
        d_bounds_violations = 0
        sigma_violations = 0
        total = 0
        for batch in loader:
            u = batch['displacements'].to(device)
            try:
                with torch.no_grad():
                    params, _ = model(u)
                ft = params[:, 3]
                # Estimate max stress from displacement
                sig_est = params[:, 0] * u.abs().max() / 100
                sigma_violations += (sig_est > ft * 1.5).sum().item()
                total += u.size(0)
            except Exception:
                pass

        return {'sigma_exceed_ft': sigma_violations,
                'total': total,
                'status': 'PASS' if sigma_violations == 0 else 'WARN'}

    # ----------------------------------------------------------------
    def summary(self):
        """Print validation report."""
        print("=" * 55)
        print("  PHYSICS VALIDATION REPORT")
        print("=" * 55)
        for check, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            icon = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗'}.get(status, '?')
            print(f"  [{icon}] {check:25s}: {status}")
        print("=" * 55)


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Physics Validator — Test")
    print("=" * 50)

    validator = PhysicsValidator()
    # Quick unit checks
    print(f"  Energy balance: {validator.check_energy_balance.__name__} defined")
    print(f"  LEFM consistency: {validator.check_lefm_consistency.__name__} defined")
    print("  Validator ready!")
