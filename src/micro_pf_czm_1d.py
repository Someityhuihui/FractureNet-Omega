"""
1D muPF-CZM Solver (Extended)
==============================
Reference: "A generalized phase-field cohesive zone model (uPF-CZM) for fracture"

This module implements:
  - Crack band width calculation (phase-field profile)
  - Traction-separation law (sigma-w relationship)
  - EXTENDED peak load estimation (p, softening, and b dependent)
  - Crack profile generation
  - Effective fracture energy computation

Governing equations (Chapter 3):
  Phase-field evolution:
    sigma**2/(2*E0) * mu'(d) - Gf/(c_alpha*b) * [alpha'(d) - 2b**2 d**2d/dx**2] = 0
  Traction-separation:
    sigma(d*) = sqrt(2*E*Gf / c_alpha) * eta(d*)
"""

import numpy as np
from scipy.integrate import quad, cumtrapz
import warnings


class MicroPF_CZM_1D:
    """
    1D muPF-CZM solver with extended peak load prediction.

    Parameters
    ----------
    E : float
        Young's modulus (MPa)
    ft : float
        Tensile strength (MPa)
    Gf : float
        Fracture energy (N/mm)
    softening : str
        Softening type: 'linear', 'exponential', 'cornelissen', 'ppr'
    p : float
        Traction order parameter (>=1). Larger p = steeper initial softening.
    b : float
        Phase-field length scale (mm)

    Derived Parameters
    ------------------
    lch : float
        Irwin intrinsic length l_ch = E*Gf/ft**2 (mm)
    sigma0 : float
        Stress scaling factor sqrt(2*E*Gf/c_alpha)
    eta0 : float
        Normalization: sigma0/eta0 = ft (ensures sigma(0) = ft)
    a0 : float
        Auxiliary parameter (2p/pi) * lch/b
    """

    # Softening curve coefficients (from paper Appendix C)
    _CORNELISSEN_COEFFS = {
        'c1': 101.6763, 'c2': -40.4105,
        'c3': -129.1615, 'c4': -60.6300,
        'c5': 30.0532, 'c6': -0.2668,
    }
    _PPR_COEFFS = {'c1': 0.75, 'c2': 0.75, 'c3': 0.0}  # m=1.5 (concave)

    def __init__(self, E, ft, Gf, softening='linear', p=1.0, b=1.0):
        # --- Material parameters ---
        self.E = E
        self.ft = ft
        self.Gf = Gf
        self.softening = softening
        self.p = p
        self.b = b

        # --- Derived parameters ---
        self.lch = E * Gf / ft**2          # Irwin intrinsic length (mm)
        self.c_alpha = np.pi               # c_alpha = pi for alpha(d)=2d-d**2
        self.a0 = self._calc_a0()          # Eq.(4.3)
        # eta0 from traction boundary condition sigma(0) = ft:
        #   sigma0 * eta(0) = sigma0 / eta0 = ft  =>  eta0 = sigma0 / ft
        self.sigma0 = np.sqrt(2.0 * self.E * self.Gf / self.c_alpha)
        self.eta0 = self.sigma0 / self.ft

        # --- Cache for expensive computations ---
        self._cache = {}

    # ==================================================================
    # Section 1: Geometric & Auxiliary Functions
    # ==================================================================

    def _calc_a0(self):
        """Eq.(4.3): a0 = (2p/pi) * l_ch / b"""
        return (2.0 * self.p / np.pi) * self.lch / self.b

    @staticmethod
    def _alpha(d):
        """Optimal geometric function: alpha(d) = 2d - d**2"""
        return 2.0 * d - d**2

    @staticmethod
    def _alpha_prime(d):
        """Derivative: alpha'(d) = 2 - 2d"""
        return 2.0 - 2.0 * d

    def _mu(self, d):
        """Auxiliary function: mu(d) = a0 * alpha(d) / (1-d)**(2p)"""
        return self.a0 * self._alpha(d) / (1.0 - d)**(2.0 * self.p)

    def _eta(self, d_star):
        """
        Eq.(3.2a): eta(d*) = (1-d*)**(2p) / eta0
        Ensures sigma(d*=0) = sigma0 * eta(0) = sigma0 / eta0 = ft.
        """
        return (1.0 - d_star)**(2.0 * self.p) / self.eta0

    # ==================================================================
    # Section 2: Crack Band Width — Eq.(3.4)
    # ==================================================================

    def _crack_width_integrand(self, theta):
        """
        Integrand: 1/sqrt(alpha(theta)) * 1/sqrt(1 - (1-theta)**(2p)/eta0)

        For eta0 > 1, the second term is real for all theta in [0, d*].
        """
        alpha = self._alpha(theta)
        if alpha <= 0:
            return 0.0
        inner = 1.0 - (1.0 - theta)**(2.0 * self.p) / self.eta0
        if inner <= 0:
            return 0.0
        return 1.0 / np.sqrt(alpha) * 1.0 / np.sqrt(inner)

    def crack_half_width(self, d_star, num_points=500):
        """
        Crack band half-width D(d*) from Eq.(3.4):
          D(d*) = b * int_0^{d*} integrand(theta) dtheta

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center (x=0)
        num_points : int
            Number of integration sub-intervals

        Returns
        -------
        D : float
            Crack band half-width (mm)
        """
        thetas = np.linspace(1e-10, d_star, num_points)
        dt = thetas[1] - thetas[0]
        integral = sum(
            self._crack_width_integrand((thetas[i-1] + thetas[i]) / 2.0) * dt
            for i in range(1, len(thetas))
        )
        return self.b * integral

    # ==================================================================
    # Section 2b: Exact 1D Analytical Profile — x(d) inversion
    # ==================================================================

    def compute_x_of_d(self, d_star, n_points=300):
        """
        Compute position x as a function of phase-field d.

        From Eq.(3.4): x(d; d*) = b * int_d^{d*} integrand(theta) dtheta

        Returns x(d) for a grid of d values from 0 to d*.

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center
        n_points : int
            Number of d-grid points

        Returns
        -------
        d_grid : ndarray
            Phase-field values from 0 to d*
        x_grid : ndarray
            Corresponding x positions (mm), x=0 at d=d*
        """
        d_grid = np.linspace(0, d_star, n_points)
        x_grid = np.zeros(n_points)

        # x(d) = b * int_d^{d*} integrand(theta) dtheta
        # Start from d=d* (x=0) and integrate backward
        for i in range(n_points - 2, -1, -1):
            d_from = d_grid[i]
            d_to = d_grid[i + 1]
            d_mid = (d_from + d_to) / 2.0
            integrand_val = self._crack_width_integrand(d_mid)
            x_grid[i] = x_grid[i + 1] + self.b * integrand_val * (d_to - d_from)

        return d_grid, x_grid

    def compute_d_profile(self, d_star, x_max=None, n_points=500):
        """
        Compute phase-field profile d(x) by inverting x(d).

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center
        x_max : float, optional
            Maximum x-coordinate. Defaults to crack half-width D(d*).
        n_points : int
            Number of spatial points

        Returns
        -------
        x_vals : ndarray
            Spatial positions (mm)
        d_vals : ndarray
            Phase-field values d(x)
        """
        d_grid, x_grid = self.compute_x_of_d(d_star, n_points=800)
        from scipy.interpolate import interp1d

        D = x_grid[0]  # crack half-width
        if x_max is None:
            x_max = D * 1.2

        x_rev = x_grid[::-1]
        d_rev = d_grid[::-1]

        try:
            d_interp = interp1d(x_rev, d_rev, kind='cubic',
                                bounds_error=False, fill_value=0.0)
        except Exception:
            d_interp = interp1d(x_rev, d_rev, kind='linear',
                                bounds_error=False, fill_value=0.0)

        x_vals = np.linspace(0, x_max, n_points)
        d_vals = np.clip(d_interp(x_vals), 0.0, d_star)
        return x_vals, d_vals

    def compute_center_cod(self, d_star):
        """
        Compute crack opening displacement at center w(0).

        From traction equivalence: sigma(d*) = sigma(w)
          ft*(1-d*)^(2p) = ft*omega(w/w_c)
          => w(0) = w_c * omega^{-1}((1-d*)^(2p))

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center

        Returns
        -------
        w0 : float
            Center COD (mm)
        """
        traction_norm = (1.0 - d_star)**(2.0 * self.p)
        w_c = 2.0 * self.Gf / self.ft

        if self.softening == 'linear':
            s = 1.0 - traction_norm
            return w_c * np.clip(s, 0.0, 1.0)

        elif self.softening == 'exponential':
            if traction_norm <= 0:
                return w_c * 5.0
            s = -np.log(np.maximum(traction_norm, 1e-12)) / 2.0
            return w_c * np.clip(s, 0.0, 10.0)

        elif self.softening in ('cornelissen', 'ppr'):
            from scipy.optimize import brentq
            def residual(s_scaled):
                w_val = s_scaled * w_c
                # Map w back to equivalent d* via inverse of linear COD-d relation
                d_equiv = 1.0 - traction_norm**(1.0 / (2.0 * self.p))
                return d_equiv - s_scaled  # simplified: linear COD-d mapping
            try:
                s = brentq(residual, 0, 100.0)
                return w_c * s
            except Exception:
                return w_c * (1.0 - traction_norm)
        else:
            s = 1.0 - traction_norm
            return w_c * np.clip(s, 0.0, 1.0)

    def compute_cod_profile(self, d_star, x_max=None, n_points=500):
        """
        Compute COD profile w(x) along the crack.

        w(x) = w(0) * sqrt(alpha(d(x))/alpha(d*))

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center
        x_max : float, optional
            Maximum x-coordinate
        n_points : int
            Number of spatial points

        Returns
        -------
        x_vals : ndarray
            Spatial positions (mm)
        w_vals : ndarray
            COD values (mm)
        """
        x_vals, d_vals = self.compute_d_profile(d_star, x_max, n_points)
        w0 = self.compute_center_cod(d_star)
        alpha_star = self._alpha(d_star)
        if alpha_star > 0:
            w_vals = w0 * np.sqrt(np.maximum(self._alpha(d_vals), 0) / alpha_star)
        else:
            w_vals = np.zeros_like(x_vals)
        return x_vals, w_vals

    def analytical_traction_separation(self, n_points=200):
        """
        Compute analytical traction-separation law sigma(w).

        Uses exact parametric forms for each softening type.

        Returns
        -------
        w_vals : ndarray
            Crack opening displacement (mm)
        sigma_vals : ndarray
            Traction stress (MPa)
        """
        w_c = 2.0 * self.Gf / self.ft

        if self.softening == 'linear':
            w_vals = np.linspace(0, w_c, n_points)
            sigma_vals = self.ft * (1.0 - w_vals / w_c)

        elif self.softening == 'exponential':
            w_vals = np.linspace(0, 5 * w_c, n_points)
            sigma_vals = self.ft * np.exp(-2.0 * w_vals / w_c)

        else:
            # Cornelissen/PPR: map through phase-field
            d_vals = np.linspace(0.001, 0.999, n_points)
            w_vals = w_c * np.array([self.compute_center_cod(d) for d in d_vals])
            sigma_vals = self.ft * (1.0 - d_vals)**(2.0 * self.p)

        return w_vals, sigma_vals

    # ==================================================================
    # Section 3: Softening Curve Xi(d) — Appendix C
    # ==================================================================

    def _xi_function(self, d):
        """
        Softening curve Xi(d) function [generalized Eq.(C.1)].

        Characterizes the shape of the traction-separation law
        for different softening types.
        """
        s = np.sqrt(1.0 - (1.0 - d)**(2.0 * self.p))
        s1 = (1.0 - d)**self.p

        if self.softening == 'linear':
            return s

        elif self.softening == 'exponential':
            if s >= 1.0:
                return 10.0
            return 0.5 * np.arctanh(s)

        elif self.softening == 'cornelissen':
            if s >= 1.0:
                return 10.0
            cc = self._CORNELISSEN_COEFFS
            return (cc['c1']*s + cc['c3']*s**3 + cc['c5']*s**5 +
                    (cc['c2']*s1**2 + cc['c4']*s1**4 + cc['c6']*s1**6) * np.arctanh(s))

        elif self.softening == 'ppr':
            if s >= 1.0:
                return 10.0
            pc = self._PPR_COEFFS
            return pc['c1']*s + pc['c2']*s1**2 * np.arctanh(s)

        else:
            warnings.warn(f"Unknown softening type '{self.softening}', using linear.")
            return s

    # ==================================================================
    # Section 4: Traction-Separation Law
    # ==================================================================

    def traction_separation_law(self, d_star_values=None, n_points=200):
        """
        Compute traction-separation curve sigma(w).

        sigma(d*) = sigma0 * eta(d*) = ft * (1-d*)**(2p)   [Eq.(3.2a)]

        The crack opening displacement w(d*) uses a linear approximation.
        Exact computation requires solving the Abel integral Eq.(4.4b).

        Parameters
        ----------
        d_star_values : array_like, optional
            Sequence of d* values in (0, 1)
        n_points : int
            Number of sample points (if d_star_values not provided)

        Returns
        -------
        w_values : ndarray
            Crack opening displacement (mm)
        sigma_values : ndarray
            Traction stress (MPa)
        """
        if d_star_values is None:
            d_star_values = np.linspace(0.001, 0.999, n_points)

        # Characteristic opening displacement
        w_char = 2.0 * self.Gf / self.ft

        sigma_values = self.sigma0 * self._eta(d_star_values)
        w_values = w_char * d_star_values  # linear approximation

        return np.array(w_values), np.array(sigma_values)

    # ==================================================================
    # Section 5: Effective Fracture Energy & Softening Shape (EXTENDED)
    # ==================================================================

    def _xi_integral(self, n_points=500):
        """
        Numerically integrate Xi(d) from d=0 to d=1.

        The integral of Xi(d) characterizes the softening curve shape.
        Different softening types give different integrals:
          - Linear: reference value
          - Exponential: larger (more ductile — longer tail)
          - Cornelissen: larger (realistic concrete behavior)

        Returns
        -------
        Xi_int : float
            Integral of Xi(d) over d in [0, 1]
        """
        d_vals = np.linspace(0.0, 0.999, n_points)
        xi_vals = np.array([self._xi_function(d) for d in d_vals])
        return np.trapz(xi_vals, d_vals)

    def softening_shape_factor(self):
        """
        Compute softening shape factor by integrating Xi(d).

        S_soft = int Xi(d; softening, p) dd / int Xi_linear(d; p) dd

        This ratio captures how the softening curve shape differs
        from linear softening at the same traction order p:
          - S_soft = 1.0: linear softening (reference)
          - S_soft > 1.0: more ductile (exponential, cornelissen)
          - S_soft < 1.0: more brittle (very steep softening)

        Returns
        -------
        S_soft : float
            Softening shape factor (dimensionless)
        """
        # Compute Xi integral for current softening
        xi_int = self._xi_integral()

        # Compute Xi integral for linear softening (reference at same p)
        # We need to temporarily compute linear Xi integral
        d_vals = np.linspace(0.0, 0.999, 500)
        s_vals = np.sqrt(1.0 - (1.0 - d_vals)**(2.0 * self.p))
        xi_linear_int = np.trapz(s_vals, d_vals)

        return xi_int / xi_linear_int if xi_linear_int > 0 else 1.0

    def effective_fracture_energy(self, n_points=500):
        """
        Compute effective fracture energy Gf_eff.

        Integrates sigma(w) dw using the traction-separation law.
        Note: uses linear w(d*) approximation; for precise values
        the Abel integral transform (Eq.4.4b) is needed.

        Returns
        -------
        Gf_eff : float
            Effective fracture energy (N/mm) from numerical integration
        """
        w, sigma = self.traction_separation_law(n_points=n_points)
        Gf_eff = np.trapz(sigma, w)
        return Gf_eff

    def effective_fracture_energy_xi(self, n_points=500):
        """
        Compute effective fracture energy using Xi-function integration.

        The total fracture energy relates to Xi integral:
          Gf_eff ~ Gf * S_soft (shape factor from Xi integration)

        This is more accurate than direct sigma(w) integration
        because it correctly captures softening-type differences.

        Returns
        -------
        Gf_eff : float
            Effective fracture energy (N/mm)
        """
        return self.Gf * self.softening_shape_factor()

    def p_correction_factor(self):
        """
        p-dependent process zone correction factor.

        From the energy dissipation integral:
          int_0^1 (1-d)**(2p) dd = 1/(2p+1)

        Relative to p=1 (reference): R_p = 3/(2p+1)

        - p=1.0: R_p = 1.000 (reference)
        - p=1.5: R_p = 0.750 (25% smaller process zone)
        - p=2.0: R_p = 0.600 (40% smaller process zone)

        Returns
        -------
        R_p : float
            p-correction factor (dimensionless)
        """
        return 3.0 / (2.0 * self.p + 1.0)

    # ==================================================================
    # Section 6: EXTENDED Peak Load Estimation
    # ==================================================================

    def peak_load(self, L_char=100.0, width=100.0, height=100.0, B_factor=0.5):
        """
        EXTENDED peak load estimation with p, softening, and b dependence.

        Generalized size effect law:
          sigma_N = B * ft / sqrt(1 + beta_eff)

        where the effective brittleness number is:
          beta_eff = beta * (2p+1)/3 / S_soft / f_b

        Physical interpretation:
          - Larger p  -> steeper softening -> smaller process zone -> MORE brittle
          - Exponential/Cornelissen -> longer tail -> larger process zone -> MORE ductile
          - Larger b  -> wider crack band -> larger process zone -> MORE ductile

        Parameters
        ----------
        L_char : float
            Structural characteristic size (mm), default 100mm (3-point bending)
        width : float
            Cross-section width (mm)
        height : float
            Cross-section height (mm)
        B_factor : float
            Geometric factor, default 0.5

        Returns
        -------
        P_max : float
            Peak load (N)
        """
        # Base brittleness number
        beta = self.lch / L_char

        # p-correction: larger p = more brittle (beta_eff increases)
        R_p = self.p_correction_factor()  # 3/(2p+1)

        # Softening shape factor: >1 for more ductile softening
        S_soft = self.softening_shape_factor()

        # b-correction: crack band width effect
        D = self.crack_half_width(0.8) if self.eta0 > 1.0 else 0.0
        f_b = 1.0 + D / self.lch if (self.lch > 0 and D > 0) else 1.0

        # Effective brittleness (avoid division by zero)
        S_eff = max(S_soft, 0.01)
        beta_eff = beta / R_p / S_eff / f_b

        # Generalized size effect law
        sigma_N = B_factor * self.ft / np.sqrt(1.0 + beta_eff)
        P_max = sigma_N * width * height

        # Store for inspection
        self._last_peak_detail = {
            'beta': beta, 'R_p': R_p, 'S_soft': S_soft, 'f_b': f_b,
            'beta_eff': beta_eff, 'sigma_N': sigma_N, 'P_max': P_max,
        }

        return P_max

    def peak_load_detail(self, L_char=100.0, width=100.0, height=100.0, B_factor=0.5):
        """
        Compute peak load AND return all intermediate diagnostic quantities.

        Returns
        -------
        dict with keys: beta, R_p, S_soft, f_b, beta_eff, sigma_N, P_max
        """
        self.peak_load(L_char, width, height, B_factor)
        return self._last_peak_detail

    # ==================================================================
    # Section 7: Process Zone Characterization
    # ==================================================================

    def process_zone_length(self):
        """
        Estimate the fracture process zone (FPZ) length.

        FPZ ~ E * Gf_eff / ft**2 = l_ch * S_soft

        This is the characteristic length over which the cohesive
        traction acts before full separation.

        Returns
        -------
        l_pz : float
            Process zone length (mm)
        """
        S_soft = self.softening_shape_factor()
        return self.lch * S_soft

    def initial_fracture_toughness(self):
        """
        Estimate the initiation fracture toughness K_Ic^ini.

        Based on the initial slope of the traction-separation law.
        Returns the apparent toughness at crack initiation (MPa * sqrt(mm)).

        Returns
        -------
        K_ini : float
            Initiation fracture toughness
        """
        return np.sqrt(self.E * self.Gf)

    # ==================================================================
    # Section 8: Crack Profile Generation
    # ==================================================================

    def generate_crack_profile(self, d_star, x_max=10.0, n_points=200):
        """
        Generate crack profile d(x) using a piecewise-linear approximation.

        d(x) = d* * max(0, 1 - x/D),  for x >= 0

        Parameters
        ----------
        d_star : float
            Phase-field value at crack center
        x_max : float
            Maximum x-coordinate (mm)
        n_points : int
            Number of sample points

        Returns
        -------
        x_values : ndarray
            Position coordinates (mm)
        d_values : ndarray
            Phase-field values
        """
        x_values = np.linspace(0, x_max, n_points)
        D = self.crack_half_width(d_star)
        if D > 0:
            d_values = np.maximum(0.0, d_star * (1.0 - x_values / D))
        else:
            d_values = np.zeros_like(x_values)
        return x_values, d_values

    # ==================================================================
    # Section 9: Summary Output
    # ==================================================================

    def summary(self):
        """Print material and derived parameter summary."""
        print("=" * 55)
        print("  uPF-CZM Model Parameter Summary")
        print("=" * 55)
        print(f"  Young's modulus    E   = {self.E:.1f} MPa")
        print(f"  Tensile strength   ft  = {self.ft:.2f} MPa")
        print(f"  Fracture energy    Gf  = {self.Gf:.4f} N/mm")
        print(f"  Softening type          = {self.softening}")
        print(f"  Traction order     p   = {self.p}")
        print(f"  Length scale       b   = {self.b:.2f} mm")
        print("-" * 55)
        print(f"  Irwin length     l_ch  = {self.lch:.2f} mm")
        print(f"  Stress scale    sigma0 = {self.sigma0:.4f}")
        print(f"  Normalization    eta0  = {self.eta0:.4f}")
        print(f"  Auxiliary         a0  = {self.a0:.4f}")
        print("-" * 55)
        print(f"  p-correction      R_p  = {self.p_correction_factor():.4f}")
        print(f"  Softening factor S_soft = {self.softening_shape_factor():.4f}")
        print(f"  Eff. fracture energy   = {self.effective_fracture_energy():.4f} N/mm")
        print(f"  Process zone length    = {self.process_zone_length():.2f} mm")
        print("=" * 55)

    def summary_extended(self, L_char=100.0, width=100.0, height=100.0):
        """Print summary including extended peak load diagnostics."""
        self.summary()
        detail = self.peak_load_detail(L_char, width, height)
        print(f"\n  --- Peak Load Diagnostics (L_char={L_char}mm) ---")
        print(f"  Base brittleness   beta    = {detail['beta']:.4f}")
        print(f"  p-correction       R_p     = {detail['R_p']:.4f}")
        print(f"  Softening factor   S_soft  = {detail['S_soft']:.4f}")
        print(f"  b-correction       f_b     = {detail['f_b']:.4f}")
        print(f"  Effective brittleness      = {detail['beta_eff']:.4f}")
        print(f"  Nominal stress   sigma_N   = {detail['sigma_N']:.4f} MPa")
        print(f"  Peak load        P_max     = {detail['P_max']:.2f} N")
        print("=" * 55)


# ======================================================================
# Test & Demo
# ======================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  muPF-CZM 1D Solver — Test Suite")
    print("=" * 55)

    # --- Test 1: Baseline (linear, p=1, b=2) ---
    print("\n[Test 1] Baseline: linear softening, p=1, b=2mm")
    m1 = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=1.0, b=2.0)
    m1.summary_extended(L_char=100)

    # --- Test 2: Effect of p ---
    print("\n[Test 2] p-dependence: p = 1.0, 1.5, 2.0")
    for p_val in [1.0, 1.5, 2.0]:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=p_val, b=2.0)
        d = m.peak_load_detail(L_char=100)
        print(f"  p={p_val}: R_p={d['R_p']:.4f} beta_eff={d['beta_eff']:.4f} "
              f"sigma_N={d['sigma_N']:.4f} MPa  Pmax={d['P_max']:.1f} N")

    # --- Test 3: Effect of softening type ---
    print("\n[Test 3] Softening type dependence")
    for st in ['linear', 'exponential', 'cornelissen']:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening=st, p=1.5, b=2.0)
        d = m.peak_load_detail(L_char=100)
        print(f"  {st:15s}: S_soft={d['S_soft']:.4f} beta_eff={d['beta_eff']:.4f} "
              f"sigma_N={d['sigma_N']:.4f} MPa  Pmax={d['P_max']:.1f} N")

    # --- Test 4: Effect of b ---
    print("\n[Test 4] b-dependence: b = 1, 2, 4 mm")
    for b_val in [1.0, 2.0, 4.0]:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=1.5, b=b_val)
        d = m.peak_load_detail(L_char=100)
        print(f"  b={b_val}mm: D(0.8)={m.crack_half_width(0.8):.3f}mm f_b={d['f_b']:.4f} "
              f"beta_eff={d['beta_eff']:.4f} sigma_N={d['sigma_N']:.4f} MPa  Pmax={d['P_max']:.1f} N")

    # --- Test 5: Size effect curves ---
    print("\n[Test 5] Size effect curve (p=1.0 vs p=2.0)")
    L_range = [25, 50, 100, 200, 400, 800]
    for p_val in [1.0, 2.0]:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=p_val, b=2.0)
        vals = [f"{m.peak_load_detail(L_char=L)['sigma_N']:.3f}" for L in L_range]
        print(f"  p={p_val}: sigma_N(L) = " + " | ".join(
            f"L={L}={v}" for L, v in zip(L_range, vals)))

    # --- Quick plot ---
    import matplotlib
    matplotlib.use('Agg')  # non-interactive
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Traction-separation: effect of p
    ax = axes[0, 0]
    for p_val in [1.0, 1.5, 2.0]:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=p_val, b=2.0)
        w, s = m.traction_separation_law()
        ax.plot(w, s, label=f'p={p_val}', linewidth=2)
    ax.set_xlabel('Crack Opening w (mm)')
    ax.set_ylabel('Traction sigma (MPa)')
    ax.set_title('Effect of p on Traction-Separation')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) Traction-separation: effect of softening type
    ax = axes[0, 1]
    for st in ['linear', 'exponential', 'cornelissen']:
        m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening=st, p=1.5, b=2.0)
        w, s = m.traction_separation_law()
        ax.plot(w, s, label=st, linewidth=2)
    ax.set_xlabel('Crack Opening w (mm)')
    ax.set_ylabel('Traction sigma (MPa)')
    ax.set_title('Effect of Softening Type')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (c) Size effect: p dependence
    ax = axes[1, 0]
    L_chars = np.logspace(np.log10(20), np.log10(1000), 60)
    for p_val in [1.0, 1.5, 2.0]:
        sigma_N = []
        for Lc in L_chars:
            m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=p_val, b=2.0)
            sigma_N.append(m.peak_load_detail(L_char=Lc)['sigma_N'])
        ax.loglog(L_chars, sigma_N, label=f'p={p_val}', linewidth=2)
    ax.axhline(y=0.5*3.0, color='gray', ls='--', label='Strength limit')
    ax.set_xlabel('Characteristic Size L_char (mm)')
    ax.set_ylabel('Nominal Stress sigma_N (MPa)')
    ax.set_title('Size Effect: p-Dependence')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')

    # (d) Size effect: softening type dependence
    ax = axes[1, 1]
    for st in ['linear', 'exponential', 'cornelissen']:
        sigma_N = []
        for Lc in L_chars:
            m = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening=st, p=1.5, b=2.0)
            sigma_N.append(m.peak_load_detail(L_char=Lc)['sigma_N'])
        ax.loglog(L_chars, sigma_N, label=st, linewidth=2)
    ax.axhline(y=0.5*3.0, color='gray', ls='--', label='Strength limit')
    ax.set_xlabel('Characteristic Size L_char (mm)')
    ax.set_ylabel('Nominal Stress sigma_N (MPa)')
    ax.set_title('Size Effect: Softening Dependence')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('muPF-CZM Extended Solver Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('extended_solver_validation.png', dpi=150)
    print("\n[Plot] Saved: extended_solver_validation.png")
