import numpy as np
from replica_figuras_2_y_3 import (
    binary_entropy, kato_a_b_upper, kato_a_b_lower,
    delta_U, delta_L, simulated_channel_data, optimize_key_rate_for_eta
)

def km_a_transmitancia(L_km, alpha_db=0.2, eta_det=1.0):
    """
    Convierte una distancia de fibra en kilómetros a transmitancia de canal (eta).
    alpha_db: Pérdida óptica en dB/km (típicamente 0.2 a 1550 nm).
    eta_det: Eficiencia de detección de los APD o SNSPD de Bob.
    """
    return eta_det * (10 ** (-(alpha_db * L_km) / 10.0))
def run_monte_carlo_qkd_protocol(N_pulses=100_000_000, eta=0.5, d=1e-9,
                                 mu_S=0.48, mu_D=0.05, mu_V=0.0,
                                 p_Z=0.88, p_S=0.8, p_D=0.1, p_V=0.1,
                                 delta_mis=0.03, f_ec=1.16,
                                 xi=71.0, eps=1e-20 / 144.0,
                                 batch_size=10_000_000, seed=42):
    rng = np.random.default_rng(seed)
    q = 1.0 - p_Z
    p_X = q

    N_Z_obs = {'S': 0, 'D': 0, 'V': 0}
    N_Z_err_obs = {'S': 0, 'D': 0, 'V': 0}
    N_X_err_obs = {'S': 0, 'D': 0, 'V': 0}
    N_Cross_obs = {'S': 0, 'D': 0, 'V': 0}

    mus_arr = np.array([mu_S, mu_D, mu_V])
    labels = ['S', 'D', 'V']

    remaining = int(N_pulses)

    while remaining > 0:
        B = min(batch_size, remaining)
        remaining -= B

        a_i = rng.integers(0, 2, size=B, dtype=np.int8)
        alpha_i = (rng.random(B) >= p_Z).astype(np.int8)
        w_idx = rng.choice(3, size=B, p=[p_S, p_D, p_V])
        mu_i = mus_arr[w_idx]

        m_i = rng.poisson(mu_i * eta)

        m_X = rng.binomial(m_i, q)
        m_Z = m_i - m_X

        dark_Z_any = rng.random(B) < (1.0 - (1.0 - d) ** 2)
        dark_X_any = rng.random(B) < (1.0 - (1.0 - d) ** 2)

        click_Z_line = (m_Z > 0) | dark_Z_any
        click_X_line = (m_X > 0) | dark_X_any

        only_Z = click_Z_line & (~click_X_line)
        only_X = click_X_line & (~click_Z_line)
        cross_click = click_Z_line & click_X_line

        for idx_w, w_name in enumerate(labels):
            mask_w = (w_idx == idx_w)

            N_Cross_obs[w_name] += int(np.sum(cross_click & mask_w))

            mask_sift_w = only_Z & (alpha_i == 0) & mask_w
            n_zw = int(np.sum(mask_sift_w))
            N_Z_obs[w_name] += n_zw

            if n_zw > 0:
                m_Z_sub = m_Z[mask_sift_w]
                err_dark_Z = np.where(m_Z_sub == 0, rng.random(n_zw) < 0.5, rng.random(n_zw) < (0.5 * d))
                err_mis_Z = rng.random(n_zw) < delta_mis
                N_Z_err_obs[w_name] += int(np.sum(err_dark_Z | err_mis_Z))

            mask_X_w = only_X & (alpha_i == 1) & mask_w
            n_xw = int(np.sum(mask_X_w))
            if n_xw > 0:
                m_X_sub = m_X[mask_X_w]
                err_dark_X = np.where(m_X_sub == 0, rng.random(n_xw) < 0.5, rng.random(n_xw) < (0.5 * d))
                err_mis_X = rng.random(n_xw) < delta_mis
                N_X_err_obs[w_name] += int(np.sum(err_dark_X | err_mis_X))

    N_Z_tilde, N_X_err_tilde, N_Cross_tilde, _, _ = simulated_channel_data(
        N_pulses, eta, mu_S, q, d, delta_mis, mu_D, mu_V, p_S, p_D, p_V, mode='passive'
    )

    p_int_1 = p_S * mu_S * np.exp(-mu_S) + p_D * mu_D * np.exp(-mu_D)
    c1 = ((1.0 - q) / q) * (p_Z / p_X)
    one_minus_d_sq = (1.0 - d) ** 2
    c2 = (p_Z * (1.0 - q) * (1.0 - 2.0 * q) * one_minus_d_sq
          / (1.0 - (q ** 2 + (1.0 - q) ** 2) * one_minus_d_sq))

    denom_decoy = mu_S - mu_D
    t_S = -(p_int_1 / p_S) * (mu_D * np.exp(mu_S)) / (mu_S * denom_decoy)
    t_D = +(p_int_1 / p_D) * (mu_S * np.exp(mu_D)) / (mu_D * denom_decoy)
    t_V = -(p_int_1 / p_V) * mu_S / (mu_D * denom_decoy)

    N_Z_1_tilde = max(0.0, t_S * N_Z_tilde['S'] + t_D * N_Z_tilde['D'] + t_V * N_Z_tilde['V'])
    N_ph_1_tilde = max(0.0, (p_int_1 / mu_D) * (
            c1 * (np.exp(mu_D) * N_X_err_tilde['D'] / p_D - N_X_err_tilde['V'] / p_V) +
            c2 * (np.exp(mu_D) * N_Cross_tilde['D'] / p_D - N_Cross_tilde['V'] / p_V)
    ))

    sqrt_N = np.sqrt(N_pulses)

    dU_ZS = delta_U(N_Z_obs['S'], N_Z_tilde['S'], N_pulses, eps)
    dL_ZD = delta_L(N_Z_obs['D'], N_Z_tilde['D'], N_pulses, eps)
    dU_ZV = delta_U(N_Z_obs['V'], N_Z_tilde['V'], N_pulses, eps)
    aU_Z1, _, b_minus_a_Z1 = kato_a_b_upper(N_pulses, N_Z_1_tilde, eps)

    N_Z_1_L = (t_S * (N_Z_obs['S'] + dU_ZS)
               + t_D * (N_Z_obs['D'] - dL_ZD)
               + t_V * (N_Z_obs['V'] + dU_ZV)
               - b_minus_a_Z1 * sqrt_N) / (1.0 + 2.0 * aU_Z1 / sqrt_N)

    dU_XD = delta_U(N_X_err_obs['D'], N_X_err_tilde['D'], N_pulses, eps)
    dL_XV = delta_L(N_X_err_obs['V'], N_X_err_tilde['V'], N_pulses, eps)
    dU_CD = delta_U(N_Cross_obs['D'], N_Cross_tilde['D'], N_pulses, eps)
    dL_CV = delta_L(N_Cross_obs['V'], N_Cross_tilde['V'], N_pulses, eps)
    aL_ph1, _, b_minus_a_ph1 = kato_a_b_lower(N_pulses, N_ph_1_tilde, eps)

    N_ph_1_U = ((p_int_1 / mu_D) * (
            c1 * (np.exp(mu_D) * (N_X_err_obs['D'] + dU_XD) / p_D - (N_X_err_obs['V'] - dL_XV) / p_V) +
            c2 * (np.exp(mu_D) * (N_Cross_obs['D'] + dU_CD) / p_D - (N_Cross_obs['V'] - dL_CV) / p_V)
    ) + b_minus_a_ph1 * sqrt_N) / (1.0 - 2.0 * aL_ph1 / sqrt_N)

    N_sift_obs = sum(N_Z_obs.values())
    N_sift_err_obs = sum(N_Z_err_obs.values())
    e_bit_obs = N_sift_err_obs / max(1, N_sift_obs)

    N_PA = (N_sift_obs - N_Z_1_L
            + N_Z_1_L * float(binary_entropy(N_ph_1_U / N_Z_1_L))
            + xi)
    N_EC = f_ec * N_sift_obs * float(binary_entropy(e_bit_obs))
    ell_final = max(0.0, N_sift_obs - N_PA - N_EC)
    R_mc = ell_final / N_pulses

    report = "================ RESULTADOS MONTE CARLO (PASOS P1-P8) ================\n"
    report += f"Simulacion de N = {N_pulses:,} pulsos (eta = {eta}, d = {d:.0e})\n"
    report += f"Longitud de clave cribada (N_sift)      : {N_sift_obs:,} bits\n"
    report += f"QBER observado en base Z (e_bit)        : {e_bit_obs * 100:.3f} %\n"
    report += f"Cota inferior monofotonica (N_Z^{{1,L}})  : {N_Z_1_L:,.1f}\n"
    report += f"Cota superior errores de fase (N_ph^{{1,U}}): {N_ph_1_U:,.1f} (tasa = {N_ph_1_U / N_Z_1_L * 100:.2f} %)\n"
    report += f"Bits sacrificados en Priv. Amp. (N_PA)  : {N_PA:,.1f} bits\n"
    report += f"Bits consumidos en Error Corr. (N_EC)   : {N_EC:,.1f} bits\n"
    report += f"Clave secreta neta generada (l)         : {ell_final:,.1f} bits\n"
    report += f"Tasa de clave secreta empirica (R = l/N): {R_mc:.6e} bits/pulso\n"
    report += "======================================================================\n"

    with open('resultados_montecarlo.txt', 'w') as f:
        f.write(report)

    print(report)
    return R_mc


if __name__ == '__main__':
    # Ejemplo: Simular un enlace urbano de 15 kilómetros
    distancia_km = 15.0
    eta_calculado = km_a_transmitancia(distancia_km)

    print(f"Simulando enlace de {distancia_km} km (Transmitancia equivalente eta = {eta_calculado:.4f})")

    run_monte_carlo_qkd_protocol(N_pulses=100_000_000, eta=eta_calculado, d=1e-9)