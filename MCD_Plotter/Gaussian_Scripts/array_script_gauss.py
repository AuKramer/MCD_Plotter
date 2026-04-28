import numpy as np

def parse_multidip_file(filename):
    """
    Parses a Multiwfn electric or magnetic dipole moment file and returns:
    - dipoles: dict with (i, j) tuple keys and [x, y, z, deltaE] values
    - states: set of all state indices found
    """
    dipoles = {}
    states = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if (not line or 
                line.startswith('i') or 
                line.lower().startswith('note') or 
                line.startswith('Transition') or 
                line.startswith('#')):
                continue
            parts = line.split()
            # Only process lines with at least 6 columns (i, j, x, y, z, deltaE)
            if len(parts) >= 6:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    deltaE = float(parts[5])
                    dipoles[(i, j)] = [x, y, z, deltaE]
                    states.update([i, j])
                except ValueError:
                    continue
    return dipoles, sorted(states)

# File names
elec_file = 'transdipmom_elec.txt'
mag_file  = 'transdipmom_mag.txt'

# Parse files
mu_elec, states_elec = parse_multidip_file(elec_file)
mu_mag,  states_mag  = parse_multidip_file(mag_file)

states = sorted(set(states_elec) | set(states_mag))
n_states = max(states) + 1

# Write electric dipole array to file 
with open('mu_elec_array.txt', 'w') as f:
    f.write('i j mu_x mu_y mu_z deltaE\n')
    for (i, j), arr in mu_elec.items():
        f.write(f'{i} {j} {arr[0]:.8f} {arr[1]:.8f} {arr[2]:.8f} {arr[3]:.8f}\n')

# Write magnetic dipole array to file 
with open('mu_mag_array.txt', 'w') as f:
    f.write('i j mx my mz deltaE\n')
    for (i, j), arr in mu_mag.items():
        f.write(f'{i} {j} {arr[0]:.8f} {arr[1]:.8f} {arr[2]:.8f} {arr[3]:.8f}\n')

print("Arrays written to mu_elec_array.txt and mu_mag_array.txt, with deltaE from the 6th column.")

