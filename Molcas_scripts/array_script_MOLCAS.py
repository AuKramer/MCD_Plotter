import numpy as np

EV_TO_AU = 1.0 / 27.211386246

def parse_multidip_file(filename, invert=False):
    """
    Parses a Multiwfn electric or magnetic dipole moment file.

    Parameters
    ----------
    filename : str
        Input text file from Multiwfn.
    invert : bool
        If True, multiply all x,y,z components by -1 on read
        (used for magnetic dipoles).

    Returns
    -------
    dipoles : dict
        Keys: (i, j) state index pairs (0-based)
        Values: [x, y, z, deltaE_au]
    states : list
        Sorted list of all state indices seen.
    """
    dipoles = {}
    states = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip headers/comments
            if (not line or 
                line.startswith('i') or 
                line.lower().startswith('note') or 
                line.startswith('Transition') or 
                line.startswith('#')):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    deltaE_ev = float(parts[5])
                    deltaE_au = deltaE_ev * EV_TO_AU

                    # --- CHANGE 1: optional inversion on read ---
                    if invert:
                        x = -x
                        y = -y
                        z = -z
                    # -------------------------------------------

                    dipoles[(i, j)] = [x, y, z, deltaE_au]
                    states.update([i, j])
                except ValueError:
                    continue
    return dipoles, sorted(states)

compound_name = "CompoundName"  # set as appropriate
elec_file = 'transdipmom_elec.txt'
mag_file  = 'transdipmom_mag.txt'

# Electric: normal read
mu_elec, states_elec = parse_multidip_file(elec_file, invert=False)

# Magnetic: SAME logic, but with sign-inverted components on read
mu_mag,  states_mag  = parse_multidip_file(mag_file, invert=True)

# Union of all states
all_states = sorted(set(states_elec) | set(states_mag))
states_shifted = [s + 1 for s in all_states]
n_states = max(states_shifted)

# Build energies from 0 -> s transitions (unchanged)
energies_au = []
for s_shift in range(1, n_states + 1):
    s = s_shift - 1
    if s == 0:
        energies_au.append(0.0)
    elif (0, s) in mu_elec:
        energies_au.append(mu_elec[(0, s)][3])
    elif (0, s) in mu_mag:
        energies_au.append(mu_mag[(0, s)][3])
    else:
        energies_au.append(0.0)

with open('energies.txt', 'w') as f:
    f.write(f"# {n_states} (atomic units)\n")
    for en in energies_au:
        f.write(f"{en:.8f}\n")

def write_dipole_files(dipole_dict, all_states, prefix, compound_name, antisymmetric=False):
    """
    Writes transition dipole/magnetic-dipole moment matrices.

    For antisymmetric=True (magnetic):
    - Forward i->j value is taken as-is from dipole_dict
      (already inverted if parse_multidip_file(..., invert=True) was used).
    - Backward j->i value is enforced as -forward (antisymmetric).
    """
    components = ['1', '2', '3']
    lines = []
    seen = set()
    for i in all_states:
        for j in all_states:
            i_shift = i + 1
            j_shift = j + 1
            if (i_shift, j_shift) in seen:
                continue

            vals_fwd = dipole_dict.get((i, j), [0.0, 0.0, 0.0, 0.0])
            lines.append((i_shift, j_shift, vals_fwd))
            seen.add((i_shift, j_shift))

            if i != j:
                if (j_shift, i_shift) not in seen:
                    if antisymmetric:
                        # backward = -forward for x,y,z, same deltaE
                        vals_bwd = [-vals_fwd[0], -vals_fwd[1], -vals_fwd[2], vals_fwd[3]]
                    else:
                        vals_bwd = dipole_dict.get((j, i), vals_fwd)
                    lines.append((j_shift, i_shift, vals_bwd))
                    seen.add((j_shift, i_shift))

    # Sort by second state, then first (unchanged)
    lines_sorted = sorted(lines, key=lambda x: (x[1], x[0]))
    for comp_idx, comp_name in enumerate(components):
        filename = f"{prefix}-{comp_name}.txt"
        with open(filename, 'w') as f:
            f.write(f"# {compound_name} i j real imag\n")
            for i_shift, j_shift, vals in lines_sorted:
                f.write(f"{i_shift} {j_shift} {vals[comp_idx]:.8f} 0.00000000\n")

# Electric dipoles: symmetric handling (no inversion, no antisymmetry)
write_dipole_files(mu_elec, all_states, 'dipole', compound_name, antisymmetric=False)

# Magnetic dipoles: values already inverted on read, antisymmetric enforced here
write_dipole_files(mu_mag,  all_states, 'angmom', compound_name, antisymmetric=True)

print("Generated energies.txt, dipole-[1..3].txt, and angmom-[1..3].txt with OM Format.")
