import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from replica_figuras_2_y_3 import compute_key_rate


def km_to_eta(L_km, alpha_db=0.2, eta_det=0.80):
    return eta_det * (10.0 ** (-(alpha_db * L_km) / 10.0))


def optimize_mu_for_fixed_q(eta, q, N, d, delta_mis, mode='passive'):
    """Optimiza mu_S > 0.05 para un valor fijo de q."""
    mu_candidates = np.linspace(0.055, 0.90, 50)
    vals = compute_key_rate(eta, mu_candidates, q, N, d, mode=mode,
                            asymptotic=False, delta_mis=delta_mis)
    best_idx = int(np.argmax(vals))
    best_val = float(vals[best_idx])
    if best_val <= 0.0:
        return 0.0, float(mu_candidates[best_idx])

    def obj(x):
        mu_val = float(x[0])
        if not (0.0501 < mu_val < 1.5):
            return 0.0
        r = compute_key_rate(eta, mu_val, q, N, d, mode=mode,
                             asymptotic=False, delta_mis=delta_mis)
        return -float(r)

    res = minimize(obj, x0=[mu_candidates[best_idx]], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-11, 'maxiter': 150})
    return max(best_val, -float(res.fun)), float(res.x[0])


def optimize_rate_free_q(eta, N, d, delta_mis, mode='passive'):
    """Optimiza conjuntamente (mu_S, q) con 0 < q < 0.5."""
    mu_grid = np.linspace(0.06, 0.85, 35)
    q_grid = np.geomspace(1e-3, 0.49, 35)
    MU_M, Q_M = np.meshgrid(mu_grid, q_grid, indexing='ij')

    R_grid = compute_key_rate(eta, MU_M, Q_M, N, d, mode=mode,
                              asymptotic=False, delta_mis=delta_mis)
    idx = np.unravel_index(np.argmax(R_grid), R_grid.shape)
    best_R = float(R_grid[idx])
    if best_R <= 0.0:
        return 0.0

    def obj(x):
        mu_val, log10_q = float(x[0]), float(x[1])
        q_val = 10.0 ** log10_q
        if not (0.0501 < mu_val < 1.5 and 1e-6 < q_val < 0.499):
            return 0.0
        r = compute_key_rate(eta, mu_val, q_val, N, d, mode=mode,
                             asymptotic=False, delta_mis=delta_mis)
        return -float(r)

    res = minimize(obj, x0=[MU_M[idx], np.log10(Q_M[idx])], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-11, 'maxiter': 200})
    return max(best_R, -float(res.fun))


def run_proposal_1(L_km=50.0, N=1e10, alpha_db=0.2, eta_det=0.80):
    eta = km_to_eta(L_km, alpha_db=alpha_db, eta_det=eta_det)
    deltas = np.linspace(0.0, 0.10, 41)

    results = {}
    lines = [
        f"PROPUESTA 1: SENSIBILIDAD A DESALINEACION DE POLARIZACION (L = {L_km} km, N = {N:.0e})\n",
        "delta_mis(%) | R_pasivo(d=1e-9) | R_activo(d=1e-9) | R_pasivo(d=1e-6) | R_activo(d=1e-6)\n"
    ]

    for d_val in [1e-9, 1e-6]:
        r_pass, r_act = [], []
        for dm in deltas:
            r_pass.append(optimize_rate_free_q(eta, N, d_val, dm, mode='passive'))
            r_act.append(optimize_rate_free_q(eta, N, d_val, dm, mode='active'))
        results[d_val] = (np.array(r_pass), np.array(r_act))

        zero_idx = np.where(results[d_val][0] <= 0.0)[0]
        thresh = deltas[zero_idx[0]] * 100.0 if len(zero_idx) > 0 else np.nan
        print(f"[d = {d_val:.0e}, L = {L_km:.0f} km] Umbral de fallo pasivo: {thresh:.2f}% de desalineación")

    for i, dm in enumerate(deltas):
        lines.append(
            f"{dm*100:11.2f} | {results[1e-9][0][i]:16.6e} | {results[1e-9][1][i]:16.6e} | "
            f"{results[1e-6][0][i]:16.6e} | {results[1e-6][1][i]:16.6e}\n"
        )

    with open('resumen_propuesta_1.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    plt.figure(figsize=(8.2, 5.4), dpi=150)
    plt.plot(deltas * 100, results[1e-9][0], '-', color='#1f77b4', lw=2.0,
             label=r'Pasivo ($d = 10^{-9}$)')
    plt.plot(deltas * 100, results[1e-9][1], '--', color='#1f77b4', lw=1.8,
             label=r'Activo ($d = 10^{-9}$)')
    plt.plot(deltas * 100, results[1e-6][0], '-', color='#d62728', lw=2.0,
             label=r'Pasivo ($d = 10^{-6}$)')
    plt.plot(deltas * 100, results[1e-6][1], '--', color='#d62728', lw=1.8,
             label=r'Activo ($d = 10^{-6}$)')
    plt.axvline(3.0, color='gray', linestyle=':', lw=1.3, label=r'Valor nominal del paper ($\delta_{\mathrm{mis}} = 3\%$)')

    plt.xlabel(r'Desalineación de polarización $\delta_{\mathrm{mis}}$ (%)', fontsize=11)
    plt.ylabel(r'Tasa de clave secreta por pulso $R = \ell / N$', fontsize=11)
    plt.title(f'Propuesta 1: Sensibilidad a desalineación ($L = {L_km:.0f}\\text{{ km}}$, $N = 10^{{10}}$)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('Propuesta_1_sensibilidad_delta_mis.png')
    plt.show()


if __name__ == '__main__':
    run_proposal_1(L_km=50.0, N=1e10)
