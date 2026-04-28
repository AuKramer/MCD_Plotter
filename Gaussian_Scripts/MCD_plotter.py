import numpy as np
import argparse
from math import pi, log

prefac= 27.21138602

def read_dipole_array_with_energies(filename):
    dipoles = {}
    states = set()
    energies = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or \
               any(line.lower().startswith(prefix) for prefix in ['i ', 'n ', 'state', 'note', 'transition']):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                i, j = int(parts[0]), int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                deltaE = float(parts[5])
                dipoles[(i, j)] = np.array([x, y, z])
                states.update([i, j])
                if i == 0 and j not in energies:
                    energies[j] = deltaE
            except ValueError:
                continue
    energies[0] = 0.0
    return dipoles, energies, sorted(states)


def calculate_B_terms_simple(mu_elec, mu_mag, energies, states, m=0, degen_threshold=1e-5):
    g0 = 1
    B_terms = {}
    B_contributions = {}
    B_contrib_details = {}

    for n in states:
        if n == m:
            continue
        
        Bn = 0.0
        contribs = {}
        term1_list = []
        term2_list = []

        # ----- Term 1 loop -----
        for k in states:
            if k == m:
                continue
            
            try:
                mu_nm = mu_elec[(n, m)]
            except KeyError:
                try:
                    mu_nm = mu_elec[(m, n)]
                except KeyError:
                    continue
            try:
                mu_km = mu_elec[(k, m)]
            except KeyError:
                try:
                    mu_km = mu_elec[(m, k)]
                except KeyError:
                    continue
            try:
                mu_kn = mu_elec[(k, n)]
            except KeyError:
                try:
                    mu_kn = mu_elec[(n, k)]
                except KeyError:
                    continue

            try:
                mag_km = mu_mag[(k, m)]
            except KeyError:
                try:
                    mag_km = -mu_mag[(m, k)]
                except KeyError:
                    continue

            delta_e_1 = energies.get(k, 0) - energies.get(m, 0)
            if abs(delta_e_1) < degen_threshold:
                continue

            cross1 = np.cross(mu_nm, mu_kn)
            term1 = np.dot(mag_km, cross1) / delta_e_1  # raw sign
            term1_prefac= term1 * prefac
            term1_list.append((k, term1_prefac))
            
            if k not in contribs:
                contribs[k] = 0.0
            contribs[k] += term1_prefac
            Bn += term1_prefac

        # ----- Term 2 loop -----
        for k in states:
            if k == n:
                continue
            
            try:
                mu_mn = mu_elec[(m, n)]
            except KeyError:
                try:
                    mu_mn = mu_elec[(n, m)]
                except KeyError:
                    continue
            try:
                mu_km = mu_elec[(k, m)]
            except KeyError:
                try:
                    mu_km = mu_elec[(m, k)]
                except KeyError:
                    continue
            try:
                mu_kn = mu_elec[(k, n)]
            except KeyError:
                try:
                    mu_kn = mu_elec[(n, k)]
                except KeyError:
                    continue

            try:
                mag_nk = mu_mag[(n, k)]
            except KeyError:
                try:
                    mag_nk = -mu_mag[(k, n)]
                except KeyError:
                    continue

            delta_e_2 = energies.get(k, 0) - energies.get(n, 0)
            if abs(delta_e_2) < degen_threshold:
                continue

            cross2 = np.cross(mu_mn, mu_km)
            term2 = np.dot(mag_nk, cross2) / delta_e_2  # raw sign
            term2_prefac=term2 *prefac
            term2_list.append((k, term2_prefac))
            
            if k not in contribs:
                contribs[k] = 0.0
            contribs[k] += term2_prefac
            Bn += term2_prefac

        # Combined raw contributions 
        contrib_list = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Final B term 
        B_terms[n] = -2.0 / 3 * g0 * Bn

        # Store raw term1/term2 lists for later printing 
        B_contributions[n] = contrib_list
        B_contrib_details[n] = {
            "term1": sorted(term1_list, key=lambda x: abs(x[1]), reverse=True),
            "term2": sorted(term2_list, key=lambda x: abs(x[1]), reverse=True)
        }

    return B_terms, B_contributions, B_contrib_details


def write_B_terms(B_terms, energies, m, filename):
    with open(filename, 'w') as f:
        f.write("#{}\n".format(len(B_terms)))
        for n, B in sorted(B_terms.items()):
            energy = energies.get(n, 0.0)
            f.write(f"{energy:.8f} {B:.8e}\n")


def write_B_contributions(B_contributions, energies, filename, top_n=5, B_contrib_details=None):
    with open(filename, 'w') as f:
        f.write(f"Top {top_n} contributing states per B term:\n\n")
        
        for n, contribs in sorted(B_contributions.items()):
            energy = energies.get(n, 0.0)
            f.write(f"State {n} Energy {energy:.6f} eV\n")

            if B_contrib_details and n in B_contrib_details:
                term1_list_raw = B_contrib_details[n]["term1"]
                term2_list_raw = B_contrib_details[n]["term2"]

                # dicts of raw values
                term1_dict_raw = dict(term1_list_raw)
                term2_dict_raw = dict(term2_list_raw)

                # --- Term 1 main top-5 section (printed with flipped sign) ---
                f.write("\n--- Term 1 contributions ---\n")
                for k, val_raw in term1_list_raw[:top_n]:
                    f.write(f"  {k}  {-val_raw:.6e}\n")

                # Additional block for Term 1 top 5, 
                f.write("\n--- Additional data for Term 1 top 5 (including Term 2 values) ---\n")
                seen_add_t1 = set()
                for k, v1_raw in term1_list_raw[:top_n]:
                    if k in seen_add_t1:
                        continue
                    seen_add_t1.add(k)
                    v2_raw = term2_dict_raw.get(k, 0.0)
                    v1 = -v1_raw
                    v2 = -v2_raw
                    vsum = v1 + v2
                    f.write(f"  {k}  Term1: {v1:.6e}  Term2: {v2:.6e}  Sum: {vsum:.6e}\n")

                # --- Term 2 main top-5 section (printed with flipped sign) ---
                f.write("\n--- Term 2 contributions ---\n")
                for k, val_raw in term2_list_raw[:top_n]:
                    f.write(f"  {k}  {-val_raw:.6e}\n")

                # Additional block for Term 2 top 5, 
                f.write("\n--- Additional data for Term 2 top 5 (including Term 1 values) ---\n")
                seen_add_t2 = set()
                for k, v2_raw in term2_list_raw[:top_n]:
                    if k in seen_add_t2:
                        continue
                    seen_add_t2.add(k)
                    v1_raw = term1_dict_raw.get(k, 0.0)
                    v1 = -v1_raw
                    v2 = -v2_raw
                    vsum = v1 + v2
                    f.write(f"  {k}  Term1: {v1:.6e}  Term2: {v2:.6e}  Sum: {vsum:.6e}\n")

            # Combined (term1 + term2) from contribs list, printed with flipped sign
            f.write("\n--- Combined (term1 + term2) contributions (from contribs list) ---\n")
            for k, val_raw in contribs[:top_n]:
                f.write(f"  {k}  {-val_raw:.6e}\n")
            f.write("\n")

        # Full contributions (all k) with flipped sign
        f.write("\nFull contributions below:\n\n")
        for n, contribs in sorted(B_contributions.items()):
            energy = energies.get(n, 0.0)
            f.write(f"State {n} Energy {energy:.6f} eV\n")
            for k, val_raw in contribs:
                f.write(f"  {k}  {-val_raw:.6e}\n")
            f.write("\n")


def main(max_state):
    mu_elec, energies, states = read_dipole_array_with_energies("mu_elec_array.txt")
    mu_mag, _, _ = read_dipole_array_with_energies("mu_mag_array.txt")
    filtered_states = [s for s in states if s <= max_state]

    B_terms, B_contributions, B_contrib_details = calculate_B_terms_simple(
        mu_elec, mu_mag, energies, filtered_states, m=0
    )

    write_B_terms(B_terms, energies, 0, "B_terms.txt")
    write_B_contributions(
        B_contributions, energies, "B_Contributions.txt", top_n=5, B_contrib_details=B_contrib_details
    )

    print(f"Calculation complete for states up to {max_state}. Outputs in B_terms.txt and B_Contributions.txt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCD B-term calculation, electric dipoles symmetric, magnetic antisymmetric")
    parser.add_argument("max_state", type=int, help="Maximum excited state to include")
    args = parser.parse_args()
    main(args.max_state)
