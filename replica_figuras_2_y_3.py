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
    Utiliza racionalización para evitar cancelación numérica cuando N es grande.
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
    dark_err_term = d * (1.0 - d) + 0.5 * (d ** 2)

    for w in ['S', 'D', 'V']:
        mu = mus[w]
        pw = probs[w]

        if mode == 'passive':
            exp_X_vac = np.exp(-mu * eta * q)
            exp_Z_vac = np.exp(-mu * eta * (1.0 - q))
            p_betaZ_given_alphaZ = exp_X_vac * one_minus_d_sq * (1.0 - exp_Z_vac * one_minus_d_sq)
            N_Z[w] = N * p_Z * pw * p_betaZ_given_alphaZ

            term_A_Z = exp_Z_vac * dark_err_term
            term_B_Z = (1.0 - exp_Z_vac) * (0.5 * d)
            p_errZ = exp_X_vac * one_minus_d_sq * (term_A_Z + term_B_Z)
            N_Z_err[w] = N * pw * p_Z * p_errZ

            p_betaX_given_alphaX = exp_Z_vac * one_minus_d_sq * (1.0 - exp_X_vac * one_minus_d_sq)
            N_X[w] = N * p_X * pw * p_betaX_given_alphaX

            term_A_X = exp_X_vac * dark_err_term
            term_B_X = (1.0 - exp_X_vac) * (0.5 * d)
            p_errX = exp_Z_vac * one_minus_d_sq * (term_A_X + term_B_X)
            N_X_err[w] = N * pw * p_X * p_errX + delta_mis * N_X[w]

            p_cross = ((1.0 - exp_Z_vac) * (1.0 - exp_X_vac)
                       + (1.0 - exp_Z_vac) * exp_X_vac * (1.0 - one_minus_d_sq)
                       + (1.0 - exp_X_vac) * exp_Z_vac * (1.0 - one_minus_d_sq)
                       + np.exp(-mu * eta) * ((1.0 - one_minus_d_sq) ** 2))
            N_Cross[w] = N * pw * p_cross

        elif mode == 'active':
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

    p_int_1 = (p_S * mu_S * np.exp(-mu_S)
               + p_D * mu_D * np.exp(-mu_D)
               + p_V * mu_V * np.exp(-mu_V))

    c1 = ((1.0 - q) / q) * (p_Z / p_X)
    if mode == 'passive':
        one_minus_d_sq = (1.0 - d) ** 2
        c2 = (p_Z * (1.0 - q) * (1.0 - 2.0 * q) * one_minus_d_sq
              / (1.0 - (q ** 2 + (1.0 - q) ** 2) * one_minus_d_sq))
    else:
        c2 = 0.0

    denom_decoy = mu_S - mu_D
    t_S = -(p_int_1 / p_S) * (mu_D * np.exp(mu_S)) / (mu_S * denom_decoy)
    t_D = +(p_int_1 / p_D) * (mu_
