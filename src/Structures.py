"""
    Define major NMR structures:
     [1] Protein
     [2] Peak
     [3] Spectrum
"""

import re
import pickle
import itertools

import numpy as np
import pandas as pd

import nmrstarlib

from common import *
from IO import *
from Measured import *
from Expected import *
from SpinSystems import *

import scipy.stats as sps


def additive_noise(atm_type, noise_model):

    stddev = NOISE_PARAMS[noise_model]['stddev']

    if atm_type not in stddev.keys():
        atm_type = atm_type[0]
    val = sps.norm.rvs() * stddev[atm_type]

    if NOISE_PARAMS[noise_model]['threshold']:
        tol = NOISE_PARAMS[noise_model]['tolerance'][atm_type]
        while val > tol or val < -tol:
            val = sps.norm.rvs() * stddev[atm_type]

    return val


class Protein(object):

    """
    Protein object for NMR assignment. Can be used for both known proteins (for
    algorithm testing) and for unknown proteins.

    Defines following fields:
        sequence : 3-letter amino acid sequence
        size     : number of amino acids in assignable sequence
        residues : pandas dataframe with known frequencies (if any)
        locs     : pandas dataframe with mean of atom chemical shifts
        scales   : pandas dataframe with standard deviation of chemical shifts
        bmrb     : defined if NMR-STAR file is provided, otherwise None

    Data for locs and scales is obtained from BMRB data and stored in chemical_shifts.dt

    Protein is setup from an NMR-STAR v3.1 file.
    If ppm_file is provided, then it is used to correct the chemical shift values
    (useful for testing against data provided by other authors).
    """

    def __init__(self, nmrstar, ppm_file=None):

        self.bmrb = None

        self.__setup_from_nmrstar(nmrstar)
        if ppm_file is not None:
            self.__correct_via_ppm(ppm_file)

    def __setup_from_nmrstar(self, nmrstar):

        # parse file
        self.bmrb = NmrStarAccessor(nmrstar)

        # useful table
        comp = self.bmrb['Entity']['Entity_comp_index']

        # sequences
        auth_id = comp['Auth_seq_ID']
        try:
            auth_id = auth_id.astype('int32')
        except ValueError:
            print 'no Auth_seq_ID, using regular ID'
            auth_id = None
        seq_id = comp['ID'].astype('int32')

        # number of assignable monomers
        if auth_id is None:
            self.size = seq_id.max()
        else:
            self.size = min(auth_id.max(), seq_id.max())

        # assignable sequence
        if auth_id is not None:
            if auth_id.max() < seq_id.max():
                self.sequence = comp['Comp_ID'].loc[auth_id > 0].as_matrix()
            elif auth_id.max() >= seq_id.max():
                self.sequence = comp['Comp_ID'].loc[seq_id > 0].as_matrix()
        else:
            self.sequence = comp['Comp_ID'].loc[seq_id > 0].as_matrix()

        # chemical shifts
        c_shifts = self.bmrb['Assigned_chem_shift_list']['Atom_chem_shift']
        if auth_id is None or auth_id.max() >= seq_id.max():
            res_no = c_shifts['Seq_ID'].astype('int32').as_matrix()
        else:
            res_no = c_shifts['Auth_seq_ID'].astype('int32').as_matrix()
        res_type = c_shifts['Comp_ID'].as_matrix()
        atm_type = c_shifts['Atom_ID'].as_matrix()
        values = c_shifts['Val'].astype('float64').as_matrix()
        self.residues = pd.DataFrame([],
                                     index=range(self.size),
                                     columns=set(atm_type))
        self.locs = pd.DataFrame([],
                                 index=range(self.size),
                                 columns=set(atm_type))
        self.scales = pd.DataFrame([],
                                   index=range(self.size),
                                   columns=set(atm_type))

        for rn, rt, at, val in zip(res_no, res_type, atm_type, values):
            self.residues.loc[rn, at] = val
            try:
                self.locs.loc[rn, at] = DATA['mean'].loc[rt, at]
                self.scales.loc[rn, at] = DATA['stddev'].loc[rt, at]
            except KeyError:
                continue

        self.sequence = res_type

    def __correct_via_ppm(self, file):

        """
        Sets up atom frequencies from ppm file.
        """

        df = pd.read_csv(file, delim_whitespace=True)
        if 'Group' in df.columns:
            # values stored under 'Spin'
            for group, atom_type, shift in zip(df.Group, df.Atom, df.Shift):
                # extract residue type to exclude side chains
                match = re.match('([A-Z_]+)([0-9]+)', group)
                if match is None:
                    continue
                res_type, residue = match.group(1), int(match.group(2))-1
                if len(res_type) > 1:
                    continue
                # fix naming convention
                if atom_type == 'HN':
                    atom_type = 'H'
                self.residues.loc[atom_type, residue] = float(shift)
        elif 'ResAtom' in df.columns:
            for resatom, shift in zip(df.ResAtom, df.Shift):
                # extract residue number and atom type from ResAtom
                match = re.match('([0-9]+).([A-Z0-9]+)', resatom)
                if match is None:
                    continue
                residue, atom_type = int(match.group(1))-1, match.group(2)
                if atom_type == 'HN':
                    atom_type = 'H'
                self.residues.loc[atom_type, residue] = float(shift)

    def __repr__(self):

        return 'Protein (ID: {}, {}) with {} residues.'.format(
            self.bmrb.id, self.bmrb.title, self.size)

    def __str__(self):

        return 'Protein (ID: {}, {}) with {} residues.'.format(
            self.bmrb.id, self.bmrb.title, self.size)


class Experiment(object):

    """
    Object representing NMR assignment experiments for simulation.

    A protein object is required to initialize the experiment. The experiment
    consists of a set of measured peak lists and expected peaks lists.

    This object is an intermediate data format, intended to provide a summary
    of the available experimental data.

    Fields:
        :measured: dictionary of spectrum names to value arrays
        :expected: dictionary of spectrum names to atom indices
        :atoms:    list of tuples (atom_type, residue_number)
    """

    def __init__(self, protein, spectra=SPECTRA.keys(),
                 make_peaks=True,
                 peak_noise_model='FLYA',
                 make_spins=True,
                 spin_noise_model='IPASS',
                 spin_scheme=SPIN_SYSTEM_ORDER,
                 spin_file=None):

        """
        Initialize experiment.
        """

        self.protein = protein
        self.sequence = protein.sequence
        self.size = protein.size
        self.atoms, self.expected = self.__get_assignable(spectra)
        if make_peaks:
            self.measured = self.__generate_measured(spectra, protein, peak_noise_model)
        else:
            self.measured = None
        if make_spins:
            if spin_file is None:
                self.spin_systems = self.__generate_spin_systems(spin_scheme, spin_noise_model)
            else:
                self.spin_systems = self.__read_spin_systems(spin_file, spin_scheme)
            self.true_spin_systems = self.__generate_spin_systems(spin_scheme, None)
        else:
            self.spin_systems = None
            self.true_spin_systems = None


    def __get_assignable(self, spectra):

        """
        Generate assignable atoms and expected peaks.
        """

        expected = {}
        polarities = {}
        residues = {}
        atoms = set()

        # determine atoms and initialize peaks
        for sp_name in spectra:
            spectrum = SPECTRA[sp_name]
            peaks = []
            pol = []
            res = []
            # run through peaks in spectrum
            for peak_pol, peak in zip(spectrum['polarities'], spectrum['peaks']):
                # create every valid peak along protein chain
                for i, residue in enumerate(self.sequence):
                    # peak in terms of (atom_type, residue_no) tuples
                    cur = [(atm, i+delta) for atm, delta in peak]
                    valid = True
                    for atm, pos in cur:
                        if pos < 0 or pos >= self.size:
                            valid = False
                    if not valid:
                        continue
                    for t in cur:
                        atoms.add(t)
                    peaks.append(cur)
                    pol.append(peak_pol)
                    res.append(cur[0][-1])
            expected[sp_name] = peaks
            polarities[sp_name] = pol
            residues[sp_name] = res

        atoms = list(atoms)
        translator = {a: i for i, a in enumerate(atoms)}

        peaks = {}
        for sp_name in spectra:
            cur = expected[sp_name]
            peaks[sp_name] = [
                [translator[atm] for atm in peak] for peak in cur]

        data = {
            'peaks': peaks,
            'residues': residues,
            'polarities': polarities,
            'sequence': self.sequence
        }

        expected = ExpectedPeaks(data=data)

        return atoms, expected

    def __generate_measured(self, spectra, protein, noise_model='FLYA'):

        """
        Simulates peaks for given spectra assuming a normal distribution.

        Available spectra are specified in common.py.
        """

        peaks = {}
        polarities = {}

        # generate measured peaks
        for sp_name in spectra:
            spectrum = []
            pol = []
            expected = self.expected[sp_name]
            for peak in expected.peaks:
                pol.append(peak.polarity)
                m_peak = []
                for atm_no in peak.value:
                    atm_type, res_no = self.atoms[atm_no]
                    try:
                        loc = protein.residues.loc[res_no, atm_type]
                    except KeyError:
                        loc = np.nan
                    m_peak.append(loc + additive_noise(atm_type, noise_model))
                if np.any(np.isnan(m_peak)):
                    continue
                spectrum.append(m_peak)
            peaks[sp_name] = np.array(spectrum)
            polarities[sp_name] = np.array(pol, dtype=np.int32)

        data = {'peaks': peaks,
                'polarities': polarities}

        return Measured(data=data)

    def __read_peaks(self):

        """
        Returns measured peaks from NMR-STAR file.
        """

        if 'Spectral_peak_list' not in self.bmrb.keys():
            return None

        peaks = {}

        spectra = self.bmrb['Spectral_peak_list']
        for spectrum in spectra:
            name = spectrum['Experiment_name']
            # check if we can interpret the experiment name
            if name not in INTERPRETER.keys():
                n_name = None
                # if not, prompt the user for a valid translation
                while n_name not in SPECTRA.keys() and n_name != '':
                    n_name = raw_input(
                        'Unknown spectra {}. '.format(name) +
                        'Provide equivalent name from list \n\n{}\n\n'.format(
                            '\n'.join(SPECTRA.keys())) +
                        'or leave blank to skip. :')
                # if no valid translation is possible, record that
                if n_name == '':
                    INTERPRETER[name] = None
                    with open('spectra.pickle', 'w') as f:
                        pickle.dump(INTERPRETER, f)
                    continue
                # if a valid translation exists, store it
                else:
                    INTERPRETER[name] = n_name
                    with open('spectra.pickle', 'w') as f:
                        pickle.dump(INTERPRETER, f)
            # apply name change to standard name
            name = INTERPRETER[name]
            if name is None:
                continue
            try:
                df1 = spectrum['Peak_general_char']
                df2 = spectrum['Peak_char']
            except KeyError:
                continue
            df = df1.merge(df2, on='Peak_ID')
            intensities = df['Intensity_val'].astype('float64').as_matrix()
            c_shifts = df['Chem_shift_val'].astype('float64').as_matrix()
            dimension = df['Spectral_dim_ID'].astype('int32').as_matrix()
            order = {k: v for k, v in zip(
                spectrum['Spectral_dim']['ID'].astype('int32').as_matrix(),
                spectrum['Spectral_dim']['Atom_type'].as_matrix())}

            # assemble peaks into correct structure (N, H, C) and determine polarities
            n_dim = len(set(dimension))
            # if 

            # peaks[name] = MeasuredPeaks(intensities, c_shifts, dimension, order)
            peaks[name] = intensities, c_shifts, dimension, order

        return peaks

    def __generate_spin_systems(self, scheme, noise_model):

        """
        Create spin systems from assigned chemical shifts.

        scheme defines the columns of the spin systems, while noise defines
        any additive gaussian noise to be added.
        """

        self.spin_scheme = scheme

        spins = []
        for i in range(self.size):
            cur = []
            for atm_type, delta in scheme:
                try:
                    value = self.protein.residues.loc[i+delta, atm_type]
                    if noise_model is not None and delta == 0:
                        value += additive_noise(atm_type, noise_model)
                except KeyError:
                    value = np.nan
                if atm_type in ('N', 'H') and np.isnan(value):
                    break
                cur.append(value)
            if len(cur) == len(scheme):
                spins.append(np.array(cur))

        data = np.array(spins)

        return SpinSystemSet(data, scheme)

    def __read_spin_systems(self, spin_file, scheme):

        """
        Reads spin systems from file.
        """

        self.spin_scheme = scheme

        spins = []

        with open(spin_file, 'r') as f:
            for l in f:
                tmp = l.split()[1:]
                current = []
                for v in tmp:
                    try:
                        current.append(float(v))
                    except ValueError:
                        current.append(np.nan)
                spins.append(np.array(current))

        data = np.array(spins)

        return SpinSystemSet(data, scheme)

    def __repr__(self):

        return 'Experiment on {}\n{}\n{}\n{}'.format(
            self.protein, self.expected, self.measured, self.spin_systems)

    def __str__(self):

        return 'Experiment on {}\n{}\n{}\n{}'.format(
            self.protein, self.expected, self.measured, self.spin_systems)
