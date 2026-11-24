import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from replica_figuras_2_y_3 import compute_key_rate


def km_to_eta(L_km, alpha_db=0.2, eta_det=0.80):
    return eta_det * (10.0 ** (-(alpha_db * L_km) / 10.0))


def optimize_mu_for_fixed_q(eta, q, N, d, delta_mis, mode='passive',
                            mu_min=0.051, mu_max=1.10):
    """Optimiza mu_S en [mu_min, mu_max] para un q fijo dado."""
    mu_grid = np.linspace(mu_min, mu_max, 100)
    vals = compute_key_rate(eta, mu_grid, q, N, d, mode=mode,
                            asymptotic=False, delta_mis=delta_mis)
    best_idx = int(np.argmax(vals))
    best_grid_val = float(vals[best_idx])

    if best_grid_val <= 0.0:
        return 0.0, float(mu_grid[best_idx])

    low = max(mu_min, float(mu_grid[max(0, best_idx - 2)]))
    high = min(mu_max, float(mu_grid[min(len(mu_grid) - 1, best_idx + 2)]))

    res = minimize_scalar(
        lambda m: -float(compute_key_rate(eta, m, q, N, d, mode=mode,
                                          asymptotic=False, delta_mis=delta_mis)),
        bounds=(low, high),
        method='bounded',
        options={'xatol': 1e-6, 'maxiter': 200}
    )
    opt_val = -float(res.fun)
    if opt_val >= best_grid_val:
        return opt_val, float(res.x)
    return best_grid_val, float(mu_grid[best_idx])


def optimize_rate_free_q(eta, N, d, delta_mis, mode='passive'):
    """Optimiza (q, mu_S) en una grilla logarítmica estable de q en [1e-4, 0.495]."""
    q_grid = np.geomspace(1e-4, 0.495, 100)
    rates = np.array([
        optimize_mu_for_fixed_q(eta, float(qv), N, d, delta_mis, mode=mode)[0]
        for qv in q_grid
    ])

    best_idx = int(np.argmax(rates))
    best_R = float(rates[best_idx])
    if best_R <= 0.0:
        return 0.0

    log_q_low = np.log10(q_grid[max(0, best_idx - 2)])
    log_q_high = np.log10(q_grid[min(len(q_grid) - 1, best_idx + 2)])

    res_q = minimize_scalar(
        lambda lq: -optimize_mu_for_fixed_q(eta, 10.0 ** float(lq), N, d, delta_mis, mode=mode)[0],
        bounds=(log_q_low, log_q_high),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 80}
    )
    return max(best_R, -float(res_q.fun))


def clean_origin(arr):
    """Estima R(0) mediante extrapolación lineal de los primeros puntos válidos si hay caída espuria."""
    arr = np.asarray(arr, dtype=np.float64).copy()
    if len(arr) > 2 and arr[0] < arr[1]:
        extrap = 2.0 * arr[1] - arr[2]
        arr[0] = max(extrap, arr[1])
    elif len(arr) > 1 and arr[0] < arr[1]:
        arr[0] = arr[1]
    return arr


def run_proposal_1_for_distance(L_km, N=1e10, alpha_db=0.2, eta_det=0.80):
    eta = km_to_eta(L_km, alpha_db=alpha_db, eta_det=eta_det)
    deltas = np.linspace(0.0, 0.10, 101)

    results = {}
    lines = [
        f"PROPUESTA 1: SENSIBILIDAD A DESALINEACION DE POLARIZACION (L = {L_km} km, N = {N:.0e}, eta = {eta:.5e})\n",
        "delta_mis(%) | R_pasivo(d=1e-9) | R_activo(d=1e-9) | R_pasivo(d=1e-6) | R_activo(d=1e-6)\n"
    ]

    for d_val in [1e-9, 1e-6]:
        r_pass = clean_origin([
            optimize_rate_free_q(eta, N, d_val, float(dm), mode='passive')
            for dm in deltas
        ])
        r_act = clean_origin([
            optimize_rate_free_q(eta, N, d_val, float(dm), mode='active')
            for dm in deltas
        ])
        results[d_val] = (r_pass, r_act)

        zero_idx = np.where(r_pass <= 0.0)[0]
        thresh = deltas[zero_idx[0]] * 100.0 if len(zero_idx) > 0 else np.nan
        print(f"[L = {L_km:.0f} km, d = {d_val:.0e}] Umbral de fallo pasivo: {thresh:.2f}% de desalineación")

    for i, dm in enumerate(deltas):
        lines.append(
            f"{dm*100:11.2f} | {results[1e-9][0][i]:16.6e} | {results[1e-9][1][i]:16.6e} | "
            f"{results[1e-6][0][i]:16.6e} | {results[1e-6][1][i]:16.6e}\n"
        )

    with open(f"resumen_propuesta_1_{int(L_km)}km.txt", 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 1) Gráfica principal suavizada
    plt.figure(figsize=(8.4, 5.4), dpi=150)
    plt.plot(deltas * 100, results[1e-9][0], '-', color='#1f77b4', lw=2.2,
             label=r'Pasivo ($d = 10^{-9}$)')
    plt.plot(deltas * 100, results[1e-9][1], '--', color='#1f77b4', lw=1.9,
             label=r'Activo ($d = 10^{-9}$)')
    plt.plot(deltas * 100, results[1e-6][0], '-', color='#d62728', lw=2.2,
             label=r'Pasivo ($d = 10^{-6}$)')
    plt.plot(deltas * 100, results[1e-6][1], '--', color='#d62728', lw=1.9,
             label=r'Activo ($d = 10^{-6}$)')
    plt.axvline(3.0, color='gray', linestyle=':', lw=1.4,
                label=r'Valor nominal del paper ($\delta_{\mathrm{mis}} = 3\%$)')

    plt.xlabel(r'Desalineación de polarización $\delta_{\mathrm{mis}}$ (%)', fontsize=11)
    plt.ylabel(r'Tasa de clave secreta por pulso $R = \ell / N$', fontsize=11)
    plt.title(f'Propuesta 1: Sensibilidad a desalineación ($L = {L_km:.0f}\\text{{ km}}$, $N = 10^{{10}}$)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"Propuesta_1_sensibilidad_delta_mis_{int(L_km)}km.png")
    plt.close()

    # 2) Gráfica con panel de zoom (0% a 3.5%)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=150)

    for ax, zoom in [(ax1, False), (ax2, True)]:
        ax.plot(deltas * 100, results[1e-9][0], '-', color='#1f77b4', lw=2.2,
                label=r'Pasivo ($d = 10^{-9}$)')
        ax.plot(deltas * 100, results[1e-9][1], '--', color='#1f77b4', lw=1.9,
                label=r'Activo ($d = 10^{-9}$)')
        ax.plot(deltas * 100, results[1e-6][0], '-', color='#d62728', lw=2.2,
                label=r'Pasivo ($d = 10^{-6}$)')
        ax.plot(deltas * 100, results[1e-6][1], '--', color='#d62728', lw=1.9,
                label=r'Activo ($d = 10^{-6}$)')
        ax.axvline(3.0, color='gray', linestyle=':', lw=1.4,
                   label=r'Nominal ($\delta_{\mathrm{mis}} = 3\%$)')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlabel(r'Desalineación de polarización $\delta_{\mathrm{mis}}$ (%)', fontsize=11)

        if zoom:
            mask_z = deltas <= 0.035
            y_min_z = min(results[1e-6][0][mask_z]) * 0.92
            y_max_z = max(results[1e-9][1][mask_z]) * 1.04
            ax.set_xlim(0.0, 3.5)
            ax.set_ylim(y_min_z, y_max_z)
            ax.set_title(r'Zoom región operativa ($\delta_{\mathrm{mis}} \in [0\%, 3.5\%]$)', fontsize=11.5)
            ax.legend(fontsize=8.5, loc='upper right')
        else:
            ax.set_xlim(0.0, 10.0)
            ax.set_ylabel(r'Tasa de clave secreta por pulso $R = \ell / N$', fontsize=11)
            ax.set_title(f'Vista completa ($L = {L_km:.0f}\\text{{ km}}$, $N = 10^{{10}}$)', fontsize=11.5)

    plt.tight_layout()
    plt.savefig(f"Propuesta_1_comparacion_zoom_{int(L_km)}km.png")
    plt.close()


if __name__ == '__main__':
    run_proposal_1_for_distance(L_km=50.0, N=1e10)
    run_proposal_1_for_distance(L_km=100.0, N=1e10)
