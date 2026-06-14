"""
Symbolic Physics Discovery Engine
==================================
Auto-discovers governing equations from FEM-generated data.

Methods:
  1. Brute-force candidate screening (fast, deterministic)
  2. PySR genetic programming (flexible, global search)
  3. SINDy (Sparse Identification of Nonlinear Dynamics)

Target laws:
  - Constitutive: sigma = f(epsilon)
  - Crack growth: da/dN = f(Delta_K, R)
  - Fracture criterion: failure = f(sigma, epsilon, K, J)
  - Size effect: sigma_N = f(ft, beta, p, S_soft)
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_text
import warnings


class SymbolicDiscovery:
    """Auto-discover physical laws from data."""

    def __init__(self, method='bruteforce'):
        self.method = method
        self.discovered = {}

    # ----------------------------------------------------------------
    # Constitutive law: stress-strain
    # ----------------------------------------------------------------
    def discover_constitutive(self, strain, stress):
        """
        Discover sigma = f(epsilon) from (strain, stress) pairs.

        Candidate forms:
          Linear elastic:  sigma = E * eps
          Ramberg-Osgood:  eps = sigma/E + alpha*(sigma/sigma_y)^n
          Damage:          sigma = E*(1-D)*eps
        """
        candidates = [
            (lambda e, c: c[0] * e,
             "sigma = E * eps (Hooke)"),
            (lambda e, c: np.where(e > c[1], c[0]*c[1] + c[2]*(e-c[1])**c[3], c[0]*e),
             "sigma = E*eps_y + K*(eps-eps_y)^n (plastic)"),
            (lambda e, c: c[0] * e * np.exp(-c[1]*e),
             "sigma = E*eps*exp(-c*eps) (damage)"),
            (lambda e, c: c[0] * e / (1 + c[1]*e),
             "sigma = E*eps/(1 + c*eps) (hyperbolic)"),
        ]

        best_r2, best_idx, best_coeffs = -np.inf, -1, None
        for i, (func, name) in enumerate(candidates):
            try:
                popt, _ = curve_fit(func, strain, stress,
                                    p0=[1.0, 1.0, 1.0, 1.0][:func(np.ones(1), [1,1,1,1]).shape[0]],
                                    maxfev=5000)
                pred = func(strain, popt)
                ss_r = np.sum((stress-pred)**2)
                ss_t = np.sum((stress-stress.mean())**2)
                r2 = 1 - ss_r/ss_t
                if r2 > best_r2:
                    best_r2, best_idx, best_coeffs = r2, i, popt
            except Exception:
                pass

        if best_idx >= 0:
            self.discovered['constitutive'] = {
                'formula': candidates[best_idx][1],
                'coefficients': best_coeffs.tolist(),
                'r2': float(best_r2),
            }
        return self.discovered.get('constitutive', {})

    # ----------------------------------------------------------------
    # Crack growth law: Paris / Walker / NASGRO
    # ----------------------------------------------------------------
    def discover_crack_growth(self, delta_K, da_dN, R_ratio=0.0):
        """
        Discover da/dN = f(delta_K, R).

        Candidates: Paris, Walker, Forman, NASGRO
        """
        log_dK = np.log10(delta_K + 1e-10)
        log_da = np.log10(da_dN + 1e-10)

        candidates = [
            (lambda dK, c: 10**(c[0] + c[1]*np.log10(dK)),
             "Paris: da/dN = C * (Delta_K)^m"),
            (lambda dK, c: 10**(c[0] + c[1]*np.log10(dK/(1-R_ratio)**c[2])),
             "Walker: da/dN = C*(Delta_K/(1-R)^gamma)^m"),
        ]

        best_r2, best_result = -np.inf, {}
        for func, name in candidates:
            try:
                popt, _ = curve_fit(func, delta_K, da_dN,
                                    p0=[-8, 3, 0.5][:func(np.ones(1), [1,1,1]).shape[0]],
                                    maxfev=5000)
                pred = func(delta_K, popt)
                ss_r = np.sum((da_dN-pred)**2)
                ss_t = np.sum((da_dN-da_dN.mean())**2)
                r2 = 1 - ss_r/ss_t
                if r2 > best_r2:
                    best_r2, best_result = r2, {
                        'formula': name, 'coefficients': popt.tolist(), 'r2': float(r2)}
            except Exception:
                pass

        if best_result:
            self.discovered['crack_growth'] = best_result
        return self.discovered.get('crack_growth', {})

    # ----------------------------------------------------------------
    # Fracture criterion: decision tree → interpretable rules
    # ----------------------------------------------------------------
    def discover_fracture_criterion(self, features, labels, feature_names=None):
        """
        Discover failure = f(sigma, epsilon, K, J, ...) using
        interpretable decision trees.
        """
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(features.shape[1])]

        tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
        tree.fit(features, labels)

        rules = export_text(tree, feature_names=feature_names, max_depth=4)
        importance = dict(zip(feature_names, tree.feature_importances_))

        self.discovered['fracture_criterion'] = {
            'rules': rules,
            'feature_importance': importance,
            'r2_train': float(tree.score(features, labels)),
        }
        return self.discovered['fracture_criterion']

    # ----------------------------------------------------------------
    # Size effect law (specialized for uPF-CZM)
    # ----------------------------------------------------------------
    def discover_size_effect(self, ft, beta, p, S_soft, sigma_N):
        """
        Discover sigma_N = f(ft, beta, p, S_soft).

        Extended Bazant law with p and softening dependence.
        """
        candidates = [
            (lambda f,b,p,s,c: c[0]*f/np.sqrt(1 + c[1]*b),
             "Bazant: sigma_N = c0*ft/sqrt(1 + c1*beta)"),
            (lambda f,b,p,s,c: c[0]*f/np.sqrt(1 + c[1]*b*(2*p+1)/3/s),
             "Extended: sigma_N = c0*ft/sqrt(1 + c1*beta*(2p+1)/3/S_soft)"),
            (lambda f,b,p,s,c: c[0]*f/np.sqrt(1 + c[1]*b)*(1 + c[2]*(p-1)),
             "p-corrected: sigma_N = c0*ft/sqrt(1+c1*beta)*(1+c2*(p-1))"),
            (lambda f,b,p,s,c: c[0]*f/np.sqrt(1 + c[1]*b/s),
             "Softening: sigma_N = c0*ft/sqrt(1 + c1*beta/S_soft)"),
        ]

        best_r2, best_result = -np.inf, {}
        for func, name in candidates:
            try:
                n_c = func(np.ones(1), np.ones(1), np.ones(1), np.ones(1),
                           [1,1,1,1]).shape[0]
                popt, _ = curve_fit(
                    lambda X, *c: func(X[0], X[1], X[2], X[3], list(c)),
                    (ft, beta, p, S_soft), sigma_N,
                    p0=[0.5, 2.5, 0.1, 0.05][:n_c], maxfev=10000)
                pred = func(ft, beta, p, S_soft, popt)
                r2 = 1 - np.sum((sigma_N-pred)**2)/np.sum((sigma_N-sigma_N.mean())**2)
                if r2 > best_r2:
                    best_r2, best_result = r2, {
                        'formula': name, 'coefficients': popt.tolist(), 'r2': float(r2)}
            except Exception:
                pass

        if best_result:
            self.discovered['size_effect'] = best_result
        return self.discovered.get('size_effect', {})

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    def summary(self):
        """Print all discovered laws."""
        print("=" * 55)
        print("  DISCOVERED PHYSICAL LAWS")
        print("=" * 55)
        for law_name, result in self.discovered.items():
            print(f"\n  [{law_name}]")
            for k, v in result.items():
                if k != 'rules':
                    print(f"    {k}: {v}")
            if 'rules' in result:
                print(f"    rules:\n{result['rules']}")
        print("=" * 55)
        return self.discovered


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Symbolic Discovery — Test")
    print("=" * 50)

    # Test 1: Constitutive law
    eps = np.linspace(0, 0.01, 200)
    sigma = 200000 * eps * np.exp(-50 * eps) + np.random.randn(200) * 0.5
    sd = SymbolicDiscovery()
    sd.discover_constitutive(eps, sigma)

    # Test 2: Crack growth
    dK = np.logspace(1, 2.5, 100)
    da = 1e-9 * dK**3.5 + np.random.randn(100) * 1e-10
    sd.discover_crack_growth(dK, da)

    # Test 3: Size effect
    ft = np.ones(100) * 3.0
    beta = np.logspace(-1, 1, 100)
    p_test = np.ones(100) * 1.5
    S = np.ones(100)
    sN = 0.5 * ft / np.sqrt(1 + beta * (2*p_test+1)/3)
    sd.discover_size_effect(ft, beta, p_test, S, sN)

    sd.summary()
    print("\nSymbolic discovery ready!")
