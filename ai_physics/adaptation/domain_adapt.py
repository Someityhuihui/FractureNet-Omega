"""
Domain Adaptation & Meta-Learning
==================================
Generalize fracture models to new geometries/materials/loads.

Methods:
  1. Adversarial Domain Adaptation (ADA): aligns source↔target features
  2. MAML: fast few-shot adaptation to new structures
  3. Fine-tuning: standard transfer learning with physics constraints
"""

import torch
import torch.nn as nn
import copy


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainDiscriminator(nn.Module):
    """Classifies which domain a feature comes from."""

    def __init__(self, feat_dim=512, n_domains=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_domains)
        )

    def forward(self, x):
        return self.net(x)


class DomainAdaptationNetwork(nn.Module):
    """
    Adversarial domain adaptation for fracture parameter identification.

    Source domain: synthetic FEM data (abundant, labeled)
    Target domain: real experimental data (scarce, unlabeled)
    """

    def __init__(self, backbone, feat_dim=512, n_domains=2):
        super().__init__()
        self.backbone = backbone
        self.discriminator = DomainDiscriminator(feat_dim, n_domains)

    def forward(self, x, return_domain=False):
        z = self.backbone.encoder(x)
        params = self.backbone.param_head(z)
        if return_domain:
            domain_logits = self.discriminator(GradientReversalLayer.apply(z, 1.0))
            return params, domain_logits
        return params


def train_domain_adaptive(model, source_loader, target_loader,
                          epochs=50, lr=1e-4, device='cpu'):
    """
    Train with adversarial domain alignment.

    Source: labeled FEM data → supervised loss
    Target: unlabeled experimental data → domain confusion loss
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    domain_loss_fn = nn.CrossEntropyLoss()
    param_loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        epoch_loss = {'sup': 0, 'domain': 0}
        n_batches = 0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # Supervised loss on source
            src_params, src_domain = model(src_x, return_domain=True)
            sup_loss = param_loss_fn(src_params, src_y)
            domain_src_loss = domain_loss_fn(
                src_domain, torch.zeros(src_x.size(0), dtype=torch.long, device=device))

            # Domain confusion on target
            _, tgt_domain = model(tgt_x, return_domain=True)
            domain_tgt_loss = domain_loss_fn(
                tgt_domain, torch.ones(tgt_x.size(0), dtype=torch.long, device=device))

            total = sup_loss + 0.5 * (domain_src_loss + domain_tgt_loss)

            opt.zero_grad(); total.backward(); opt.step()
            epoch_loss['sup'] += sup_loss.item()
            epoch_loss['domain'] += (domain_src_loss + domain_tgt_loss).item() / 2
            n_batches += 1

        history.append({k: v / n_batches for k, v in epoch_loss.items()})
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: sup={history[-1]['sup']:.4f}, "
                  f"domain={history[-1]['domain']:.4f}")

    return model, history


# ================================================================
# MAML: Model-Agnostic Meta-Learning
# ================================================================
class MAMLAdapter:
    """
    Fast adaptation to new structures with few samples (5-10).

    Inner loop: task-specific SGD
    Outer loop: meta-update across tasks
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def adapt(self, support_set, query_set=None):
        """
        Fast adaptation to a new task.

        Args:
            support_set: list of (x, y) tuples, few-shot examples
            query_set: list of (x, y) for evaluation (optional)
        Returns:
            adapted_model, support_loss, query_loss
        """
        adapted = copy.deepcopy(self.model)

        # Inner loop: fast gradient steps
        opt_inner = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        support_losses = []
        for _ in range(self.inner_steps):
            total_loss = 0
            for x, y in support_set:
                pred, _ = adapted(x.unsqueeze(0))
                loss = nn.functional.mse_loss(pred, y.unsqueeze(0))
                total_loss += loss
            opt_inner.zero_grad(); total_loss.backward(); opt_inner.step()
            support_losses.append(total_loss.item() / len(support_set))

        # Evaluate on query set
        query_loss = 0.0
        if query_set:
            for x, y in query_set:
                with torch.no_grad():
                    pred, _ = adapted(x.unsqueeze(0))
                    query_loss += nn.functional.mse_loss(
                        pred, y.unsqueeze(0)).item()
            query_loss /= len(query_set)

        return adapted, support_losses[-1], query_loss

    def meta_train(self, task_generator, n_tasks=10, tasks_per_batch=4):
        """
        Meta-training across many tasks.

        task_generator: function that yields (support_set, query_set)
        """
        for iteration in range(n_tasks):
            meta_loss = 0.0
            for _ in range(tasks_per_batch):
                support, query = task_generator()
                _, _, q_loss = self.adapt(support, query)
                meta_loss += q_loss

            meta_loss /= tasks_per_batch
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            if iteration % 5 == 0:
                print(f"Meta-iter {iteration:3d}: query_loss={meta_loss.item():.4f}")

        return self.model


# ================================================================
# Fine-tuning with physics constraints
# ================================================================
def fine_tune_physics(model, target_loader, epochs=30, lr=1e-5, device='cpu'):
    """
    Standard fine-tuning on target domain with physics constraints.

    - Lower learning rate preserves learned features
    - Physics loss prevents catastrophic forgetting
    """
    from identification.param_identifier import PhysicsLoss

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    phys_loss = PhysicsLoss()

    for epoch in range(epochs):
        total_l = 0.0
        for batch in target_loader:
            x, y = batch['displacements'].to(device), batch['params'].to(device)
            pred, _ = model(x)
            loss = phys_loss(x, pred, x, y)
            opt.zero_grad(); loss['total'].backward(); opt.step()
            total_l += loss['total'].item()
        if epoch % 10 == 0:
            print(f"Fine-tune epoch {epoch:3d}: loss={total_l/len(target_loader):.4f}")

    return model


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Domain Adaptation — Test")
    print("=" * 50)

    # Test gradient reversal
    x = torch.randn(4, 512, requires_grad=True)
    grl = GradientReversalLayer.apply(x, 1.0)
    print(f"GRL shape: {grl.shape}, same as input: {(grl == x).all().item()}")

    # Test discriminator
    disc = DomainDiscriminator(512, 3)
    out = disc(x)
    print(f"Discriminator output: {out.shape} (should be [4,3])")

    print("Domain adaptation module ready!")
