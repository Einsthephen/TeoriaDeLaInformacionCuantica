import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from propuesta_1_desalineacion import km_to_eta, optimize_mu_for_fixed_q


def run_proposal_2(N=1e10, alpha_db=0.2, eta_det=0.80, delta_mis=0.03):
    # Malla de alta resolución para mapa de calor y curva q_opt(L) suaves
    distances_km = np.linspace(5.0, 180.0, 120)
    q_values = np.linspace(0.005, 0.495, 120)

    for d_val in [1e-9, 1e-6]:
        R_map = np.zeros((len(q_values), len(distances_km)))
        q_opt_list, r_opt_list = [], []

        for j, dist in enumerate(distances_km):
            eta = km_to_eta(dist, alpha_db=alpha_db, eta_det=eta_det)

            for i, q_val in enumerate(q_values):
                r_val, _ = optimize_mu_for_fixed_q(eta, float(q_val), N, d_val,
                                                    delta_mis, mode='passive')
                R_map[i, j] = r_val

            col_rates = R_map[:, j]
            best_i = int(np.argmax(col_rates))
            best_r = float(col_rates[best_i])

            if best_r <= 0.0:
                q_opt_list.append(np.nan)
                r_opt_list.append(0.0)
            else:
                # Refinamiento continuo en q para eliminar escalones discretos en q_opt(L)
                q_low = float(q_values[max(0, best_i - 2)])
                q_high = float(q_values[min(len(q_values) - 1, best_i + 2)])
                res_q = minimize_scalar(
                    lambda qv: -optimize_mu_for_fixed_q(eta, float(qv), N, d_val,
                                                        delta_mis, mode='passive')[0],
                    bounds=(q_low, q_high),
                    method='bounded',
                    options={'xatol': 1e-4, 'maxiter': 60}
                )
                if -float(res_q.fun) >= best_r:
                    q_opt_list.append(float(res_q.x))
                    r_opt_list.append(-float(res_q.fun))
                else:
                    q_opt_list.append(float(q_values[best_i]))
                    r_opt_list.append(best_r)

        q_opt_arr = np.array(q_opt_list)
        r_opt_arr = np.array(r_opt_list)
        log10_R = np.where(R_map > 0.0, np.log10(R_map), np.nan)

        tag = f"d_{d_val:.0e}".replace('+', '')
        lines = [
            f"PROPUESTA 2: OPTIMIZACION TOPOLOGICA DE q VS DISTANCIA (d = {d_val:.0e}, N = {N:.0e})\n",
            "Distancia(km) | q_optimo | Division_Z_X | R_maximo\n"
        ]
        for dist, qo, ro in zip(distances_km, q_opt_arr, r_opt_arr):
            if np.isnan(qo):
                lines.append(f"{dist:13.2f} | {'N/A':8s} | {'N/A':12s} | {0.0:10.6e}\n")
            else:
                lines.append(
                    f"{dist:13.2f} | {qo:8.4f} | {(1-qo)*100:5.1f}%:{qo*100:4.1f}% | {ro:10.6e}\n"
                )
        with open(f'resumen_propuesta_2_{tag}.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)

        plt.figure(figsize=(8.6, 5.4), dpi=150)
        mesh = plt.pcolormesh(distances_km, q_values, log10_R, shading='gouraud', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\log_{10}(R)$ (Tasa de clave secreta por pulso)', fontsize=10)

        valid_mask = ~np.isnan(q_opt_arr)
        plt.plot(distances_km[valid_mask], q_opt_arr[valid_mask], '-',
                 color='white', lw=2.2, label=r'$q_{\mathrm{ópt}}(L)$')

        plt.xlabel(r'Distancia de fibra $L$ (km) [$\alpha = 0.2\text{ dB/km}$, $\eta_{\mathrm{det}} = 0.80$]', fontsize=11)
        plt.ylabel(r'Transmitancia física del divisor de haz $q$ (línea $X$)', fontsize=11)
        plt.title(f'Propuesta 2: Optimización topológica de $q$ vs. distancia ($d = {d_val:.0e}$, $N = 10^{{10}}$)', fontsize=12)
        plt.legend(loc='upper right', fontsize=9.5)
        plt.tight_layout()
        plt.savefig(f'Propuesta_2_mapa_q_distancia_{tag}.png')
        plt.close()


if __name__ == '__main__':
    run_proposal_2(N=1e10)
