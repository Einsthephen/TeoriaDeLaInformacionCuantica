import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# =============================================================================
# 1. FUNCIONES MATEMÁTICAS Y DESIGUALDAD DE KATO (SECCIÓN V Y APÉNDICE A)
# =============================================================================

def binary_entropy(x):
    """
    Función de entropía binaria h(x) definida en la Ec. (6).
    """
    x = np.asarray(x, dtype=np.float64)
    res = np.ones_like(x)
    mask_zero = (x <= 0.0)
    mask_mid = (x > 0.0) & (x < 0.5)
    res[mask_zero] = 0.0
    xm = x[mask_mid]
    res[mask_mid] = -xm * np.log2(xm) - (1.0 - xm) * np.log2(1.0 - xm)
    return res


def kato_a_b_upper(N, C, eps):
    """
    Calcula a_U(N, C, eps), b_U(N, C, eps) y (b_U - a_U) según Ecs. (A2) y (A3).
    Utiliza racionalización para evitar cancelación catastrófica cuando N = 10^12.
    """
    C = np.clip(np.asarray(C, dtype=np.float64), 0.0, N)
    ln_eps = np.log(eps)  # Negativo
    sqrt_N = np.sqrt(N)

    term_C = 9.0 * C * (N - C) - 2.0 * N * ln_eps
    rad = np.maximum(0.0, - (N ** 2) * ln_eps * term_C)

    num = (216.0 * sqrt_N * C * (N - C) * ln_eps
           - 48.0 * (N ** 1.5) * (ln_eps ** 2)
           + 27.0 * np.sqrt(2.0) * (N - 2.0 * C) * np.sqrt(rad))
    den = 4.0 * (9.0 * N - 8.0 * ln_eps) * term_C

    a_U = num / den
    delta = 0.5 * ((1.0 + (4.0 * a_U) / (3.0 * sqrt_N)) ** 2) * (-ln_eps)
    b_U = np.sqrt(a_U ** 2 + delta)

    # Si a_U >= 0, racionalizamos b_U - a_U = delta / (b_U + a_U)
    b_minus_a = np.where(a_U >= 0.0, delta / (b_U + a_U + 1e-300), b_U - a_U)
    return a_U, b_U, b_minus_a


def kato_a_b_lower(N, C, eps):
    """
    Calcula a_L(N, C, eps), b_L(N, C, eps) y (b_L - a_L) según Ecs. (A7) y (A8).
    """
    C = np.clip(np.asarray(C, dtype=np.float64), 0.0, N)
    ln_eps = np.log(eps)  # Negativo
    sqrt_N = np.sqrt(N)

    term_C = 9.0 * C * (N - C) - 2.0 * N * ln_eps
    rad = np.maximum(0.0, - (N ** 2) * ln_eps * term_C)

    num = (-216.0 * sqrt_N * C * (N - C) * ln_eps
           + 48.0 * (N ** 1.5) * (ln_eps ** 2)
           + 27.0 * np.sqrt(2.0) * (N - 2.0 * C) * np.sqrt(rad))
    den = 4.0 * (9.0 * N - 8.0 * ln_eps) * term_C

    a_L = num / den
    delta = 0.5 * ((1.0 - (4.0 * a_L) / (3.0 * sqrt_N)) ** 2) * (-ln_eps)
    b_L = np.sqrt(a_L ** 2 + delta)

    b_minus_a = np.where(a_L >= 0.0, delta / (b_L + a_L + 1e-300), b_L - a_L)
    return a_L, b_L, b_minus_a


def delta_U(M, M_tilde, N, eps):
    """Término de desviación superior Delta_U de la Ec. (21)."""
    a_U, _, b_minus_a = kato_a_b_upper(N, M_tilde, eps)
    sqrt_N = np.sqrt(N)
    return b_minus_a * sqrt_N + (2.0 * a_U * M) / sqrt_N


def delta_L(M, M_tilde, N, eps):
    """Término de desviación inferior Delta_L de la Ec. (22)."""
    a_L, _, b_minus_a = kato_a_b_lower(N, M_tilde, eps)
    sqrt_N = np.sqrt(N)
    return b_minus_a * sqrt_N + (2.0 * a_L * M) / sqrt_N


# =============================================================================
# 2. MODELO DE CANAL DE PÉRDIDA LINEAL (APÉNDICE D Y REFERENCIA ACTIVA [9])
# =============================================================================

def simulated_channel_data(N, eta, mu_S, q, d, delta_mis=0.03,
                           mu_D=0.05, mu_V=0.0,
                           p_S=0.8, p_D=0.1, p_V=0.1, mode='passive'):
    """
    Calcula los conteos observables esperados bajo el modelo de canal lineal
    (Apéndice D para 'passive', y su contraparte activa de Ref. [9] para 'active').
    """
    p_Z = 1.0 - q
    p_X = q
    mus = {'S': mu_S, 'D': mu_D, 'V': mu_V}
    probs = {'S': p_S, 'D': p_D, 'V': p_V}

    N_Z = {}
    N_Z_err = {}
    N_X = {}
    N_X_err = {}
    N_Cross = {}

    one_minus_d_sq = (1.0 - d) ** 2
    dark_err_term = d * (1.0 - d) + 0.5 * (d ** 2)  # Termino A en Ecs. (D5) y (D9)

    for w in ['S', 'D', 'V']:
        mu = mus[w]
        pw = probs[w]

        if mode == 'passive':
            # --- D.1: Cálculo de N_{Z,w} (Ecs. D1 y D2) ---
            exp_X_vac = np.exp(-mu * eta * q)
            exp_Z_vac = np.exp(-mu * eta * (1.0 - q))
            p_betaZ_given_alphaZ = exp_X_vac * one_minus_d_sq * (1.0 - exp_Z_vac * one_minus_d_sq)
            N_Z[w] = N * p_Z * pw * p_betaZ_given_alphaZ

            # --- D.2: Cálculo de errores en base Z (Ecs. D4 y D5) ---
            term_A_Z = exp_Z_vac * dark_err_term
            term_B_Z = (1.0 - exp_Z_vac) * (0.5 * d)
            p_errZ = exp_X_vac * one_minus_d_sq * (term_A_Z + term_B_Z)
            N_Z_err[w] = N * pw * p_Z * p_errZ

            # --- D.3: Cálculo de N_{X,w}^{Error} (Ecs. D6 a D9) ---
            p_betaX_given_alphaX = exp_Z_vac * one_minus_d_sq * (1.0 - exp_X_vac * one_minus_d_sq)
            N_X[w] = N * p_X * pw * p_betaX_given_alphaX

            term_A_X = exp_X_vac * dark_err_term
            term_B_X = (1.0 - exp_X_vac) * (0.5 * d)
            p_errX = exp_Z_vac * one_minus_d_sq * (term_A_X + term_B_X)
            N_X_err[w] = N * pw * p_X * p_errX + delta_mis * N_X[w]

            # --- D.4: Cálculo de N_w^{Cross} (Ecs. D10 a D12) ---
            p_cross = ((1.0 - exp_Z_vac) * (1.0 - exp_X_vac)
                       + (1.0 - exp_Z_vac) * exp_X_vac * (1.0 - one_minus_d_sq)
                       + (1.0 - exp_X_vac) * exp_Z_vac * (1.0 - one_minus_d_sq)
                       + np.exp(-mu * eta) * ((1.0 - one_minus_d_sq) ** 2))
            N_Cross[w] = N * pw * p_cross

        elif mode == 'active':
            # En medición activa, Bob elige Z con prob (1-q) y X con prob q activamente.
            # Todo el pulso entra a la base seleccionada y no hay clics cruzados.
            exp_full = np.exp(-mu * eta)
            p_det_given_basis = 1.0 - exp_full * one_minus_d_sq
            p_err_given_basis = exp_full * dark_err_term + (1.0 - exp_full) * (0.5 * d)

            N_Z[w] = N * p_Z * (1.0 - q) * pw * p_det_given_basis
            N_Z_err[w] = N * pw * p_Z * (1.0 - q) * p_err_given_basis

            N_X[w] = N * p_X * q * pw * p_det_given_basis
            N_X_err[w] = N * pw * p_X * q * p_err_given_basis + delta_mis * N_X[w]
            N_Cross[w] = np.zeros_like(mu_S * q)

    N_sift = N_Z['S'] + N_Z['D'] + N_Z['V']
    N_sift_err = N_Z_err['S'] + N_Z_err['D'] + N_Z_err['V']
    e_bit = (N_sift_err / np.maximum(N_sift, 1e-300)) + delta_mis

    return N_Z, N_X_err, N_Cross, N_sift, e_bit


# =============================================================================
# 3. EVALUACIÓN DE TASA DE CLAVE SECRETA (TEOREMAS 1 Y 2, EC. 32)
# =============================================================================

def compute_key_rate(eta, mu_S, q, N, d, mode='passive', asymptotic=False,
                     mu_D=0.05, mu_V=0.0, p_S=0.8, p_D=0.1, p_V=0.1,
                     delta_mis=0.03, f_ec=1.16, xi=71.0, eps=1e-20 / 144.0):
    """
    Calcula la tasa de clave secreta por pulso R = l / N (Ec. 32).
    """
    mu_S = np.asarray(mu_S, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_Z = 1.0 - q
    p_X = q

    # Probabilidad de emitir 1 fotón p^{int}(1)
    p_int_1 = (p_S * mu_S * np.exp(-mu_S)
               + p_D * mu_D * np.exp(-mu_D)
               + p_V * mu_V * np.exp(-mu_V))

    # Constantes c1 y c2 de la Ec. (24)
    c1 = ((1.0 - q) / q) * (p_Z / p_X)
    if mode == 'passive':
        one_minus_d_sq = (1.0 - d) ** 2
        c2 = (p_Z * (1.0 - q) * (1.0 - 2.0 * q) * one_minus_d_sq
              / (1.0 - (q ** 2 + (1.0 - q) ** 2) * one_minus_d_sq))
    else:
        c2 = 0.0  # Como indica el texto tras el Teorema 1 para el caso activo

    # Coeficientes decoy t_S, t_D, t_V de la Ec. (30)
    denom_decoy = mu_S - mu_D
    t_S = -(p_int_1 / p_S) * (mu_D * np.exp(mu_S)) / (mu_S * denom_decoy)
    t_D = +(p_int_1 / p_D) * (mu_S * np.exp(mu_D)) / (mu_D * denom_decoy)
    t_V = -(p_int_1 / p_V) * mu_S / (mu_D * denom_decoy)

    N_eff = 1.0 if asymptotic else N
    N_Z, N_X_err, N_Cross, N_sift, e_bit = simulated_channel_data(
        N_eff, eta, mu_S, q, d, delta_mis, mu_D, mu_V, p_S, p_D, p_V, mode=mode
    )

    if asymptotic:
        # Límite asintótico (N -> inf): desviaciones de Kato y xi/N tienden a 0
        N_Z_1_L = t_S * N_Z['S'] + t_D * N_Z['D'] + t_V * N_Z['V']

        term_err = (np.exp(mu_D) * N_X_err['D'] / p_D) - (N_X_err['V'] / p_V)
        term_crs = (np.exp(mu_D) * N_Cross['D'] / p_D) - (N_Cross['V'] / p_V)
        N_ph_1_U = (p_int_1 / mu_D) * (c1 * term_err + c2 * term_crs)

        valid = (N_Z_1_L > 1e-30) & (N_ph_1_U >= 0.0)
        ph_rate = np.where(valid, N_ph_1_U / np.maximum(N_Z_1_L, 1e-30), 1.0)

        R = np.where(
            valid,
            N_Z_1_L * (1.0 - binary_entropy(ph_rate)) - f_ec * N_sift * binary_entropy(e_bit),
            0.0
        )
        return np.maximum(0.0, R)

    # --- Régimen de clave finita (Teoremas 1 y 2) ---
    sqrt_N = np.sqrt(N)

    # Estimaciones previas (tilde) iguales a las medias esperadas del modelo
    N_Z_tilde_1 = np.maximum(0.0, t_S * N_Z['S'] + t_D * N_Z['D'] + t_V * N_Z['V'])
    term_err_tilde = (np.exp(mu_D) * N_X_err['D'] / p_D) - (N_X_err['V'] / p_V)
    term_crs_tilde = (np.exp(mu_D) * N_Cross['D'] / p_D) - (N_Cross['V'] / p_V)
    N_ph_tilde_1 = np.maximum(0.0, (p_int_1 / mu_D) * (c1 * term_err_tilde + c2 * term_crs_tilde))

    # Teorema 2: Cota inferior N_Z^{1,L} (Ec. 25)
    dU_ZS = delta_U(N_Z['S'], N_Z['S'], N, eps)
    dL_ZD = delta_L(N_Z['D'], N_Z['D'], N, eps)
    dU_ZV = delta_U(N_Z['V'], N_Z['V'], N, eps)

    aU_Z1, _, b_minus_a_Z1 = kato_a_b_upper(N, N_Z_tilde_1, eps)
    curly_Z = (t_S * (N_Z['S'] + dU_ZS)
               + t_D * (N_Z['D'] - dL_ZD)
               + t_V * (N_Z['V'] + dU_ZV)
               - b_minus_a_Z1 * sqrt_N)

    pref_Z = 1.0 + (2.0 * aU_Z1) / sqrt_N
    N_Z_1_L = np.where(aU_Z1 > -0.5 * sqrt_N, curly_Z / pref_Z, 0.0)

    # Teorema 1: Cota superior N_ph^{1,U} (Ec. 23)
    dU_XD = delta_U(N_X_err['D'], N_X_err['D'], N, eps)
    dL_XV = delta_L(N_X_err['V'], N_X_err['V'], N, eps)
    term_1 = c1 * ((np.exp(mu_D) * (N_X_err['D'] + dU_XD) / p_D)
                   - ((N_X_err['V'] - dL_XV) / p_V))

    if mode == 'passive':
        dU_CD = delta_U(N_Cross['D'], N_Cross['D'], N, eps)
        dL_CV = delta_L(N_Cross['V'], N_Cross['V'], N, eps)
        term_2 = c2 * ((np.exp(mu_D) * (N_Cross['D'] + dU_CD) / p_D)
                       - ((N_Cross['V'] - dL_CV) / p_V))
    else:
        term_2 = 0.0

    aL_ph1, _, b_minus_a_ph1 = kato_a_b_lower(N, N_ph_tilde_1, eps)
    curly_ph = (p_int_1 / mu_D) * (term_1 + term_2) + b_minus_a_ph1 * sqrt_N
    pref_ph = 1.0 - (2.0 * aL_ph1) / sqrt_N
    N_ph_1_U = np.where(aL_ph1 < 0.5 * sqrt_N, curly_ph / pref_ph, float(N))

    # Longitud de clave secreta l (Ec. 32) y tasa R = l / N
    valid = (N_Z_1_L > 1e-12) & (N_ph_1_U >= 0.0)
    ph_rate = np.where(valid, N_ph_1_U / np.maximum(N_Z_1_L, 1e-12), 1.0)

    N_EC = f_ec * N_sift * binary_entropy(e_bit)
    ell = np.where(
        valid,
        N_Z_1_L * (1.0 - binary_entropy(ph_rate)) - xi - N_EC,
        0.0
    )
    return np.maximum(0.0, ell / N)


# =============================================================================
# 4. OPTIMIZADOR GLOBAL + LOCAL SOBRE (mu_S, p_Z = 1 - q)
# =============================================================================

# Malla precalculada para búsqueda global vectorizada instantánea
MU_GRID = np.linspace(0.055, 0.85, 65)
Q_GRID = np.geomspace(1e-4, 0.495, 65)
MU_MESH, Q_MESH = np.meshgrid(MU_GRID, Q_GRID, indexing='ij')


def optimize_key_rate_for_eta(eta, N, d, mode='passive', asymptotic=False):
    """
    Optimiza R respecto a mu_S > 0.05 y 0 < q < 0.5 (es decir, p_Z = 1 - q > 0.5).
    Paso 1: Evaluación vectorizada en malla 2D para localizar la cuenca global.
    Paso 2: Refinamiento de alta precisión con Nelder-Mead.
    """
    R_grid = compute_key_rate(eta, MU_MESH, Q_MESH, N, d, mode=mode, asymptotic=asymptotic)
    max_idx = np.unravel_index(np.argmax(R_grid), R_grid.shape)
    best_R_grid = R_grid[max_idx]

    if best_R_grid <= 0.0:
        return 0.0

    mu_init = MU_MESH[max_idx]
    q_init = Q_MESH[max_idx]

    def obj(x):
        mu_val, log10_q = x
        q_val = 10.0 ** log10_q
        if not (0.0501 < mu_val < 1.5 and 1e-7 < q_val < 0.4999):
            return 0.0
        val = compute_key_rate(eta, mu_val, q_val, N, d, mode=mode, asymptotic=asymptotic)
        return -float(val)

    res = minimize(
        obj,
        x0=[mu_init, np.log10(q_init)],
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-11, 'maxiter': 250}
    )
    return max(best_R_grid, -res.fun)


# =============================================================================
# 5. GENERACIÓN DE LAS FIGURAS 2 Y 3 DEL ARTÍCULO
# =============================================================================

def generate_paper_figures():
    configs = [
        {
            'fig_num': 2,
            'd': 1e-9,
            'ylim': (1e-9, 1e0),
            'filename': 'Figura_2_replica_d_1e-9.png',
            'title': r'Réplica Figura 2: $d = 10^{-9}$ (Pasivo: sólida, Activo: discontinua)'
        },
        {
            'fig_num': 3,
            'd': 1e-6,
            'ylim': (1e-8, 1e0),
            'filename': 'Figura_3_replica_d_1e-6.png',
            'title': r'Réplica Figura 3: $d = 10^{-6}$ (Pasivo: sólida, Activo: discontinua)'
        }
    ]

    curves_spec = [
        {'label': r'$N = 10^8$', 'N': 1e8, 'asymp': False, 'color': '#FFA500'},  # Naranja
        {'label': r'$N = 10^{10}$', 'N': 1e10, 'asymp': False, 'color': '#00FF00'},  # Verde
        {'label': r'$N = 10^{12}$', 'N': 1e12, 'asymp': False, 'color': '#FF0000'},  # Rojo
        {'label': 'Asintótico', 'N': None, 'asymp': True, 'color': '#000000'}  # Negro
    ]

    eta_vals = np.logspace(-5, 0, 140)

    for cfg in configs:
        d = cfg['d']
        print(f"\n--- Calculando Figura {cfg['fig_num']} (d = {d:.0e}) ---")
        plt.figure(figsize=(8, 5.8), dpi=150)

        for spec in curves_spec:
            N_val = spec['N']
            asymp = spec['asymp']
            color = spec['color']
            label = spec['label']

            R_passive = np.array([
                optimize_key_rate_for_eta(eta, N_val, d, mode='passive', asymptotic=asymp)
                for eta in eta_vals
            ])
            R_active = np.array([
                optimize_key_rate_for_eta(eta, N_val, d, mode='active', asymptotic=asymp)
                for eta in eta_vals
            ])

            # Enmascarar ceros para gráfica log-log limpia
            R_pas_plot = np.where(R_passive > 0, R_passive, np.nan)
            R_act_plot = np.where(R_active > 0, R_active, np.nan)

            plt.plot(eta_vals, R_pas_plot, linestyle='-', color=color, linewidth=1.8,
                     label=f"{label} (Pasivo)")
            plt.plot(eta_vals, R_act_plot, linestyle='--', color=color, linewidth=1.8,
                     label=f"{label} (Activo)")
            print(f"  Completada curva {label}")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-5, 1e0)
        plt.ylim(*cfg['ylim'])
        plt.xlabel(r'Channel transmission $\eta$', fontsize=12)
        plt.ylabel('Key rate per pulse', fontsize=12)
        plt.title(cfg['title'], fontsize=12)
        plt.grid(True, which='both', linestyle=':', alpha=0.4)
        plt.legend(fontsize=8.5, ncol=2, loc='lower right')
        plt.tight_layout()
        plt.savefig(cfg['filename'])
        print(f"Figura guardada en: {cfg['filename']}")

    plt.show()


if __name__ == '__main__':
    generate_paper_figures()