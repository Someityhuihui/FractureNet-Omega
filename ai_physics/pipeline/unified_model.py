"""
Unified Fracture AI Model — Master Pipeline
=============================================
End-to-end integration of the complete AI-physics chain.

Flow:
  1. Dataset Generation  (FEM parameterized + augmentation)
  2. Parameter ID        (CNN encoder → material + crack params)
  3. Symbolic Discovery  (auto-discover constitutive/growth laws)
  4. Domain Adaptation   (synthetic → real data)
  5. Validation          (physics consistency + energy balance)
  6. Inference           (displacement field → full fracture state)
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai_physics.identification.param_identifier import (
    PhysicsInformedParameterIdentifier, PhysicsLoss,
    train_identifier, identify_parameters, PARAM_NAMES
)
from ai_physics.discovery.symbolic_laws import SymbolicDiscovery
from ai_physics.adaptation.domain_adapt import (
    DomainAdaptationNetwork, train_domain_adaptive, MAMLAdapter
)


class UnifiedFractureModel:
    """
    Complete fracture AI system.

    Trained on FEM-generated data, adapted to real experiments,
    capable of parameter identification, physics discovery, and
    generalization to new structures.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.identifier = PhysicsInformedParameterIdentifier(in_channels=3)
        self.domain_adapter = None
        self.discovery = SymbolicDiscovery()
        self.discovered_laws = {}
        self.training_history = {}

    # ----------------------------------------------------------------
    # Stage 1: Train on synthetic data
    # ----------------------------------------------------------------
    def train_on_synthetic(self, dataloader, epochs=100):
        """Train parameter identifier on FEM-generated synthetic data."""
        print(f"[Stage 1] Training on synthetic data ({epochs} epochs)...")
        self.identifier, hist = train_identifier(
            self.identifier, dataloader, epochs=epochs, device=self.device)
        self.training_history['synthetic'] = hist
        return self.identifier

    # ----------------------------------------------------------------
    # Stage 2: Discover physical laws
    # ----------------------------------------------------------------
    def discover_laws(self, strain_data=None, stress_data=None,
                      dK_data=None, da_dN_data=None,
                      ft_data=None, beta_data=None, p_data=None,
                      S_soft_data=None, sigma_N_data=None):
        """Run symbolic discovery on accumulated data."""
        print("[Stage 2] Discovering physical laws...")

        if strain_data is not None and stress_data is not None:
            self.discovery.discover_constitutive(strain_data, stress_data)

        if dK_data is not None and da_dN_data is not None:
            self.discovery.discover_crack_growth(dK_data, da_dN_data)

        if all(v is not None for v in [ft_data, beta_data, p_data,
                                        S_soft_data, sigma_N_data]):
            self.discovery.discover_size_effect(
                ft_data, beta_data, p_data, S_soft_data, sigma_N_data)

        self.discovered_laws = self.discovery.discovered
        self.discovery.summary()
        return self.discovered_laws

    # ----------------------------------------------------------------
    # Stage 3: Adapt to real data
    # ----------------------------------------------------------------
    def adapt_to_real(self, source_loader, target_loader, epochs=50,
                      method='adversarial'):
        """
        Adapt model from synthetic domain to real experimental data.

        Methods:
          - 'adversarial': gradient reversal domain confusion
          - 'finetune': standard fine-tuning with physics constraints
          - 'maml': meta-learning for rapid few-shot adaptation
        """
        print(f"[Stage 3] Domain adaptation ({method}, {epochs} epochs)...")

        self.domain_adapter = DomainAdaptationNetwork(self.identifier)

        if method == 'adversarial':
            self.domain_adapter, hist = train_domain_adaptive(
                self.domain_adapter, source_loader, target_loader,
                epochs=epochs, device=self.device)
        else:
            from ai_physics.adaptation.domain_adapt import fine_tune_physics
            self.identifier = fine_tune_physics(
                self.identifier, target_loader, epochs=epochs, device=self.device)
            hist = []

        self.training_history['adaptation'] = hist
        return self.domain_adapter or self.identifier

    # ----------------------------------------------------------------
    # Stage 4: Validate physics consistency
    # ----------------------------------------------------------------
    def validate(self, test_loader):
        """Validate model against physics constraints."""
        print("[Stage 4] Physics validation...")
        from ai_physics.validation.physics_validator import PhysicsValidator
        validator = PhysicsValidator()
        results = validator.validate(self.identifier, test_loader, self.device)
        print(f"  Energy balance error: {results.get('energy_balance', 'N/A')}")
        print(f"  Path independence:   {results.get('path_independence', 'N/A')}")
        print(f"  LEFM consistency:    {results.get('lefm_consistency', 'N/A')}")
        return results

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def predict(self, displacement_field):
        """
        Full inference pipeline:
        displacement field → material params + crack state + predictions.
        """
        if not isinstance(displacement_field, torch.Tensor):
            displacement_field = torch.tensor(displacement_field, dtype=torch.float32)

        params = identify_parameters(
            self.identifier, displacement_field, self.device)

        # Use discovered laws for predictions
        if 'crack_growth' in self.discovered_laws:
            c = self.discovered_laws['crack_growth']['coefficients']
            K = params['stress_intensity_K']
            da_dN = 10**(c[0] + c[1]*np.log10(max(K, 1e-10)))
            params['predicted_da_dN'] = float(da_dN)

        if 'size_effect' in self.discovered_laws:
            c = self.discovered_laws['size_effect']['coefficients']
            params['predicted_sigma_N'] = float(
                c[0] * params['tensile_strength'] /
                np.sqrt(1 + c[1] * params['crack_length'] / 100))

        return params

    # ----------------------------------------------------------------
    # Save/Load
    # ----------------------------------------------------------------
    def save(self, path):
        """Save complete model state."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.identifier.state_dict(),
                   os.path.join(path, 'identifier.pt'))
        with open(os.path.join(path, 'discovered_laws.json'), 'w') as f:
            json.dump(self.discovered_laws, f, indent=2)
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model state."""
        self.identifier.load_state_dict(
            torch.load(os.path.join(path, 'identifier.pt'),
                       map_location=self.device))
        if os.path.exists(os.path.join(path, 'discovered_laws.json')):
            with open(os.path.join(path, 'discovered_laws.json')) as f:
                self.discovered_laws = json.load(f)
        print(f"Model loaded from {path}")


# ================================================================
# End-to-end training pipeline
# ================================================================
def run_full_pipeline(config=None, device='cpu'):
    """
    Execute the complete AI-physics chain.

    Args:
        config: dict with keys:
            - data_dir: path to generated dataset
            - epochs_synthetic: int (default 100)
            - epochs_adapt: int (default 50)
            - output_dir: model save path
    """
    if config is None:
        config = {}

    print("=" * 60)
    print("  FractureNet-Omega AI-Physics Pipeline")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Initialize
    model = UnifiedFractureModel(device=device)

    # Stage 1: Train on synthetic
    # (User provides dataloader from dataset generation module)
    # model.train_on_synthetic(synthetic_loader, config.get('epochs_synthetic', 100))

    # Stage 2: Discover laws
    model.discover_laws(
        strain_data=np.random.randn(200) * 0.001,
        stress_data=np.random.randn(200) * 10,
        ft_data=np.ones(100) * 3.0,
        beta_data=np.logspace(-1, 1, 100),
        p_data=np.ones(100) * 1.5,
        S_soft_data=np.ones(100),
        sigma_N_data=0.5 * 3.0 / np.sqrt(1 + np.logspace(-1, 1, 100)),
    )

    # Save
    output = config.get('output_dir', 'models/unified')
    model.save(output)

    print(f"\nPipeline complete. Model saved to {output}")
    return model


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Unified Fracture AI Model — Test")
    print("=" * 55)

    model = UnifiedFractureModel(device='cpu')

    # Test inference with dummy input
    x = torch.randn(1, 3, 128, 128)
    result = model.predict(x)
    print(f"\nInference result:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k:30s}: { {kk: f'{vv:.4f}' for kk, vv in list(v.items())[:3]} }")

    print("\nUnified model ready!")
