"""Chapter 2: match BG beam size to the corresponding LG beam."""

from configs.chapter_2 import (
    BG_MATCHING_Q,
    BG_WINDOW_ALPHA,
    BG_WINDOW_RING,
    TOTAL_DISTANCE,
    W0_LG,
    WAVELENGTH,
)
from src.beams import (
    bg_computational_window,
    bg_second_moment_radius,
    calculate_bg_parameters,
    lg_second_moment_radius,
)

CHARGES = (1, 2, 3)


def run() -> list[dict[str, float | int]]:
    results = []
    for charge in CHARGES:
        target_radius = lg_second_moment_radius(charge, W0_LG)
        w0_bg, kr = calculate_bg_parameters(charge, W0_LG, BG_MATCHING_Q)
        bg_radius = bg_second_moment_radius(charge, kr, w0_bg)
        l_rings, l_gaussian, required_window = bg_computational_window(
            charge=charge,
            kr=kr,
            w0=w0_bg,
            wavelength=WAVELENGTH,
            z=TOTAL_DISTANCE,
            alpha=BG_WINDOW_ALPHA,
            n_ring=BG_WINDOW_RING,
        )
        results.append({
            "charge": charge,
            "W_LG": target_radius,
            "w0_BG": w0_bg,
            "kr": kr,
            "W_BG": bg_radius,
            "relative_error": abs(bg_radius - target_radius) / target_radius,
            "L_rings": l_rings,
            "L_gaussian": l_gaussian,
            "L_required": required_window,
        })
    return results


def print_results(results: list[dict[str, float | int]]) -> None:
    header = (
        f"{'ell':>4}{'W_LG [m]':>14}{'w0_BG [m]':>14}{'kr [1/m]':>14}"
        f"{'W_BG [m]':>14}{'rel. error':>14}{'L req. [m]':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{int(row['charge']):>4d}"
            f"{float(row['W_LG']):>14.8f}"
            f"{float(row['w0_BG']):>14.8f}"
            f"{float(row['kr']):>14.4f}"
            f"{float(row['W_BG']):>14.8f}"
            f"{float(row['relative_error']):>14.3e}"
            f"{float(row['L_required']):>14.6f}"
        )


if __name__ == "__main__":
    print("BG-LG second-moment matching")
    print(f"q = kr*w0_BG = {BG_MATCHING_Q}")
    print(f"Window evaluated at z = {TOTAL_DISTANCE:.1f} m\n")
    print_results(run())
