#!/usr/bin/env python3
""" spec-gen.py -- A simple spectral convolution program for IR, ECD, VCD, and CPL
Usage:
    spec-gen.py [options]
        (-i <file.spec> | --input <file.spec>)
        (-g <sigma> | --gaussian <sigma> | -l <gamma> | --lorentzian <gamma>)

Positional Arguments:
    -i, --input <file.spec>             A spectrum file, see README for definition
    -g, --gaussian <sigma>              A sigma value for the broadened gaussian, see README
    -l, --lorentzian <gamma>            A gamma value for the broadened lorentzian, see README

Options:
    -h, --help                          Print this screen and exit
    -v, --version                       Print version and exit
    -q, --quadrature <points>           How fine of grid used for broadening spectrum, see README
                                            [default=1000]   
    -p, --padding <factor>              How much should you pad the energy range, see README
                                            [default=5]
"""    


#####################
# Some Improvements #
#####################
#   1) Define the output file using command line option
#       a. Allows us to have matching log file
#       b. Can deal with errors by printing to command line
#   2) Always use '.spec' ending
#   3) Use Exceptions to handle errors
#   4) Clean up code! We can probably write some functions to help
#   5) Add proper comments!


########################
# Function Definitions #
# Gaussian and Lorentzian Functions are FWHM #
########################
from math import exp, pi, sqrt, log

def gauss(sigma, x, x0):
    return (1.0 / (sigma * sqrt(2.0 * pi))) * exp(-(x - x0) ** 2.0 / (2.0 * sigma ** 2.0))

def lorentz(gamma, x, x0):
    return gamma / (2.0 * pi * ((x - x0) ** 2.0 + (gamma / 2.0) ** 2.0))

def gauss_derivative(sigma, x, x0):
    return - (x - x0) / (sigma**3 * sqrt(2.0 * pi)) * exp(-(x - x0)**2 / (2.0 * sigma**2))

def abs_int(x):
    return 3.48e-5

def cd_int(x):
    return 22.97e0 / x

def em_int(x):
    return NotImplementedError


#####################
# Necessary Imports #
#####################
from docopt import docopt
from numpy import arange
from os.path import isfile
from sys import exit


#############
# Main Code #
#############
try:
    # Our favorite option parser :)
    arguments = docopt(__doc__, version='spec-gen.py version 0.0.3')

    # Check user input and set values
    if arguments['--gaussian']:
        try:
            broaden = float(arguments['--gaussian'])
        except ValueError as err:
            exit('--> Error: {}\nHow to Fix: Gaussian sigma value should be a floating point number.'.format(err))
    if arguments['--lorentzian']:
        try:
            broaden = float(arguments['--lorentzian'])
        except ValueError as err:
            exit('--> Error: {}\nHow to Fix: Lorentzian gamma value should be a floating point number.'.format(err))
    if arguments['--quadrature']:
        try:
            num_points = int(arguments['--quadrature'])
        except ValueError as err:
            exit('--> Error: {}\nHow to Fix: Quadrature value should be an integer number.'.format(err))
    else:
        num_points = 1000
    if arguments['--padding']:
        try:
            factor = int(arguments['--padding'])
            padding = broaden * factor
        except ValueError as err:
            exit('--> Error: {}\nHow to Fix: Padding value should be an integer number.'.format(err))
    else:
        factor = 5
        padding = broaden * factor
    
    # Check input file exists
    if not isfile(arguments['--input']):
        exit('--> Error: {} does not exist?'.format(arguments['--input']))

    with open(arguments['--input'], 'r') as f:
        line = f.readline()
        sp = line.split()
        num_exc = int(sp[0].replace('#',''))
        type_spec = sp[1].lower()

        energies = [None] * num_exc
        impulses = [None] * num_exc
        for i in range(num_exc):
            sp = f.readline().split()
            energies[i] = float(sp[0])
            impulses[i] = float(sp[1])

    energy_range = [min(energies) - padding, max(energies) + padding]
    delta_e = (energy_range[1] - energy_range[0]) / (num_points)

    for_int = 0.0
    imp_val = 0.0
    integral = 0.0

    if type_spec in ['abs', 'ir']:
        if arguments['--gaussian']:
            for i in arange(energy_range[0], energy_range[1] + delta_e, delta_e):
                imp_val = 0.0
                for j in range(num_exc):
                    for_int = impulses[j] * gauss(broaden, i, energies[j]) / abs_int(i)
                    imp_val += for_int
                    integral += for_int
                print(('{} {}'.format(i, imp_val)))
        elif arguments['--lorentzian']:
            for i in arange(energy_range[0], energy_range[1] + delta_e, delta_e):
                imp_val = 0.0
                for j in range(num_exc):
                    for_int = impulses[j] * lorentz(broaden, i, energies[j]) / abs_int(i)
                    imp_val += for_int
                    integral += for_int
                print(('{} {}'.format(i, imp_val)))

    elif type_spec in ['cd', 'ecd', 'vcd']:
        if arguments['--gaussian']:
            for i in arange(energy_range[0], energy_range[1] + delta_e, delta_e):
                imp_val = 0.0
                for j in range(num_exc):
                    for_int = impulses[j] * gauss(broaden, i, energies[j]) / cd_int(i)
                    imp_val += for_int
                    integral += for_int / energies[j]
                print(('{} {}'.format(i, imp_val)))
        elif arguments['--lorentzian']:
            for i in arange(energy_range[0], energy_range[1] + delta_e, delta_e):
                imp_val = 0.0
                for j in range(num_exc):
                    for_int = impulses[j] * lorentz(broaden, i, energies[j]) / cd_int(i)
                    imp_val += for_int
                    integral += for_int / energies[j]
                print(('{} {}'.format(i, imp_val)))

    elif type_spec in ['mcda', 'mcdb']:
        energies = []
        intensities = []
        with open(arguments['--input'], 'r') as f:
            header = f.readline()
            for line in f:
                if line.strip().startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                energies.append(float(parts[0]))
                intensities.append(float(parts[1]))

        if len(energies) == 0 or len(intensities) == 0:
            exit(f"Error: No valid {type_spec.upper()} data found in input file.")

        energy_range = [min(energies) - padding, max(energies) + padding]
        delta_e = (energy_range[1] - energy_range[0]) / (num_points)

        for omega in arange(energy_range[0], energy_range[1] + delta_e, delta_e):
            val = 0.0
            prefactor = (pi / (4500*log(10) )) * 10000 * 0.001480279
            for e, inten in zip(energies, intensities):
                if arguments['--gaussian']:
                    if type_spec == 'mcda':
                        val += (inten * prefactor)* omega * gauss_derivative(broaden, omega, e)
                    else:
                        val += (inten * prefactor)* omega * gauss(broaden, omega, e)
                else:
                    val += (inten * prefactor) * lorentz(broaden, omega, e)
            print(f"{omega} {val}")

    elif type_spec == 'mcd':
        exit("Please specify the MCD term for broadening (mcda or mcdb)")

    elif type_spec in ['em', 'cpl']:
        exit("Error: Emission spectra are currently not supported")

    else:
        exit("Error: This program doesn't know how to broaden the input file")

    with open('spec-gen.log', 'w') as f:
        f.write('This program was written by Barry Moore (moore0557@gmail.com)\n')
        f.write('-> Please email with any questions/comments/suggestions\n')
        f.write('-> User is responsible for any and all broadened spectra!\n')
        f.write(' === DEBUGGING INFORMATION BELOW THIS LINE === \n')
        f.write('Input File: {}\nSpectrum Type: {}\n'.format(arguments['--input'], type_spec))
        if arguments['--gaussian']:
            f.write('Broadened with Gaussian functions, sigma = {}\n'.format(broaden))
        elif arguments['--lorentzian']:
            f.write('Broadened with Lorentzian functions, gamma = {}\n'.format(broaden))
        f.write('Number of points in broadened spectrum: {}\nEnergy padding factor: {}\n'.format(num_points, factor))
        if type_spec in ['abs', 'ir']:
            integral = integral * abs_int(1.0) * delta_e / sum(impulses)
        elif type_spec in ['cd', 'ecd', 'vcd']:
            integral = integral * cd_int(1.0) * delta_e / sum(impulses)
        f.write('Integral of broadened spectrum: {} (Should be 1.00)\n'.format(integral))

except KeyboardInterrupt:
    exit('Interrupt Detected... Exiting')
