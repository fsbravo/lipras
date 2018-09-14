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

import scipy.stats as sps


def additive_noise(atm_type, model='FLYA'):

    std = NOISE_PARAMS[model]['std']
    if atm_type not in std.keys():
        atm_type = atm_type[0]
    val = sps.norm.rvs() * std[atm_type]
    if NOISE_PARAMS[model]['threshold']:
        tol = NOISE_PARAMS[model]['tolerance'][atm_type]
        while val > tol or val < -tol:
            val = sps.norm.rvs() * std[atm_type]
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
    """

    def __init__(self, nmrstar):

        self.bmrb = None

        self.__setup_from_nmrstar(nmrstar)

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

    def __init__(self, protein, spectra=SPECTRA.keys(), noise_model='FLYA'):

        """
        Initialize experiment.
        """

        self.sequence = protein.sequence
        self.size = protein.size
        self.atoms, self.expected = self.__get_assignable(spectra)
        self.measured = self.__generate_measured(spectra, protein, noise_model)

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


class TrueProtein(object):

    """
    Assigned protein for simulation purposes.

    Description of fields
        sequence : sequence of aminoacids (3-letter code)
        size     : number of aminoacids
        residues : Pandas dataframe (columns are atom codes and rows are
                   0-indexed residue numbers)
    """

    def __init__(self, file, ppm_file=None):

        """
        Generate Protein from NMRStar file.

        If ppm_file is not None, sets up residue values from ppm_file instead
        of the NMRStar file. This is used to test against data provided by
        IPASS authors.
        """

        with open(file, 'r') as f:
            # find the sequence
            properties = [line.strip() for line in f if
                          re.findall('_Entity_comp_index.*', line)]
            f.seek(0)
            for line in f:
                if re.findall(properties[-1], line):
                    break
            self.sequence = []
            auth_seq_ids = []
            for line in f:
                if re.findall('stop_', line):
                    break
                vals = line.split()
                if len(vals) == 6:
                    self.sequence.append(vals[2])
                    try:
                        auth_seq_ids.append(int(vals[1]))
                    except ValueError:
                        auth_seq_ids.append(1)

            true_shift = -np.min(np.array(auth_seq_ids)) + 1
            safe = ('4027', '4144', '4288', '4302', '4316', '4318',
                    '4353', '4391', '4579', '4670', '4752', '4929')
            for num in safe:
                if file.endswith(num + '.txt'):
                    true_shift = 0
            shift = 0 if true_shift < 0 else true_shift
            # correct for mismatch between spin system data and NMRStar data
            self.sequence = self.sequence[shift:]
            self.size = len(self.sequence)
            print self.size

            # find frequencies
            f.seek(0)
            properties = [line.strip() for line in f if
                          re.findall('_Atom_chem_shift.*', line)]
            names = [p.split('.')[-1] for p in properties]
            f.seek(0)
            for line in f:
                if re.findall(properties[-1], line):
                    break
            df = pd.DataFrame([], columns=names)
            for line in f:
                data = line.split()
                if len(data) == len(properties):
                    df.loc[df.shape[0]] = data

        # convert numeric columns to int / float
        for c in df.columns:
            if c in ('Val', 'Val_err'):
                try:
                    df[c] = df[c].apply(float)
                except ValueError:
                    pass
            else:
                try:
                    df[c] = df[c].apply(int)
                except:
                    pass
        # separate by Atom_ID
        atom_types = set(df['Atom_ID'])
        n_residues = self.size

        self.residues = pd.DataFrame([], columns=range(n_residues),
                                     index=atom_types)

        # setup residues
        if ppm_file is None:
            for i in range(df.shape[0]):
                atom_id = df.loc[i, 'Atom_ID']
                try:
                    seq_id = int(df.loc[i, 'Auth_seq_ID']) - 1
                except ValueError:
                    seq_id = int(df.loc[i, 'Seq_ID']) - 1
                value = float(df.loc[i, 'Val'])
                self.residues.loc[atom_id, seq_id] = value
        else:
            self._setup_from_ppm(ppm_file)

        if true_shift < 0:
            self.residues.drop(range(-true_shift), axis=1, inplace=True)
            n_residues = min(df['Auth_seq_ID'].max(), df['Seq_ID'].max())
            self.residues.columns = range(n_residues)

        self.locs = pd.DataFrame([], columns=range(n_residues),
                                 index=list(atom_types))
        self.scales = pd.DataFrame([], columns=range(n_residues),
                                   index=list(atom_types))

        print n_residues
        for i in range(n_residues):
            aa = self.sequence[i]
            for j in atom_types:
                try:
                    loc = DATA['mean'].loc[aa, j]
                    self.locs.loc[j, i] = loc
                    scale = DATA['stddev'].loc[aa, j]
                    self.scales.loc[j, i] = scale
                except Exception:
                    pass

    def _setup_from_ppm(self, file):

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

    def atom_frequency(self, atom):

        """
        Takes atom as tuple (atom_type, residue) and returns frequency.
        """

        assert(len(atom) == 2)
        return self.residues.loc[atom[0], atom[1]]


class FakeProtein(object):

    """
    Protein with randomly generated values for simulation purposes.

    Description of fields
        sequence : sequence of aminoacids (3-letter code)
        size     : number of aminoacids
        residues : Pandas dataframe (columns are atom codes and rows are
                   0-indexed residue numbers)
    """

    def __init__(self, sequence, spectra):

        """
        Generate protein from collected statistics.
        """

        # if sequence is provided in one letter code format
        if len(sequence[0]) == 1:
            self.sequence = one_to_three(sequence)
        else:
            self.sequence = sequence

        self.size = len(self.sequence)

        n_residues = len(self.sequence)
        atom_types = set()
        for s in spectra:
            spectrum = SPECTRA[s]
            for p in spectrum['peaks']:
                for a in p:
                    atom_types.add(a[0])
        residues = pd.DataFrame([], columns=range(n_residues),
                                index=list(atom_types))
        locs = pd.DataFrame([], columns=range(n_residues),
                            index=list(atom_types))
        scales = pd.DataFrame([], columns=range(n_residues),
                              index=list(atom_types))

        for i in xrange(n_residues):
            aa = self.sequence[i]
            for j in atom_types:
                loc = DATA['mean'].loc[aa, j]
                locs.loc[j, i] = loc
                scale = DATA['stddev'].loc[aa, j]
                scales.loc[j, i] = scale

                if np.isnan(loc) or np.isnan(scale):
                    val = np.nan
                else:
                    val = np.random.normal(loc=DATA['mean'].loc[aa, j],
                                           scale=DATA['stddev'].loc[aa, j])
                residues.loc[j, i] = val

        self.residues = residues
        self.locs = locs
        self.scales = scales

    def atom_frequency(self, atom):

        """
        Takes atom as tuple (atom_type, residue) and returns frequency.
        """

        assert(len(atom) == 2)
        return self.residues.loc[atom[0], atom[1]]


class SimulatedExperiment(object):

    """
    A simulated experiment including measured peaks, expected peaks, and
    a set of assignable atoms
    """

    def __init__(self, protein, spectra):

        self.protein = protein
        self.spectra = spectra
        self.set_measured_peaks()
        self.set_expected_peaks()   # also initializes self.atoms
        self.set_spin_systems()     # sets clean spin systems
        self.true_x = np.array(
            [self.protein.atom_frequency(a) for a in self.atoms])

    def set_measured_peaks(self):

        """
        Simulate set of measured peaks.

        TODO: add noisy peaks, remove true peaks at random
        """
        self.measured_peaks = {}
        self.m_polarities = {}

        noise = np.array([PARAMS['std']['N'],
                          PARAMS['std']['H'],
                          PARAMS['std']['C']])
        tolerance = np.array([PARAMS['tolerance']['N'],
                              PARAMS['tolerance']['H'],
                              PARAMS['tolerance']['C']])

        for sname in self.spectra:
            peaks = []
            polarities = []
            spectrum = SPECTRA[sname]
            dim = len(spectrum['peaks'][0])
            for peak, polarity in zip(spectrum['peaks'],
                                      spectrum['polarities']):
                for j in xrange(self.protein.size):
                    atom_types, positions = zip(*peak)
                    positions = np.array(positions) + j
                    if np.any(positions < 0):
                        continue
                    values = np.array([self.protein.residues.loc[a, p] for
                                       a, p in zip(atom_types, positions)])
                    if np.any(np.isnan(values)):
                        continue
                    # option 1
                    # values += np.random.randn(dim) * noise[:dim]
                    # option 2 (same as FLYA)
                    wobble = np.random.randn(dim) * noise[:dim]
                    for i in range(dim):
                        while np.abs(wobble[i]) > tolerance[i]:
                            wobble[i] = np.random.randn() * noise[i]
                    values += wobble
                    peaks.append(values)
                    polarities.append(polarity)
            self.measured_peaks[sname] = np.stack(peaks)
            self.m_polarities[sname] = polarities

    def set_expected_peaks(self):

        """
        Get expected set of peaks.
        """

        expected_peaks = {}
        self.e_polarities = {}
        self.residues = {}

        for sname in self.spectra:
            peaks = []
            polarities = []
            residues = []
            spectrum = SPECTRA[sname]
            for peak, polarity in zip(spectrum['peaks'],
                                      spectrum['polarities']):
                for j in xrange(self.protein.size):
                    atom_types, positions = zip(*peak)
                    positions = np.array(positions) + j
                    if np.any(positions < 0):
                        continue
                    aas = [self.protein.sequence[pos] for
                           pos in positions]
                    mus = [DATA['mean'].loc[aa, atype]
                           for aa, atype in zip(aas, atom_types)]
                    # check if data exists for this aminoacid
                    if np.any(np.isnan(mus)):
                        continue
                    peaks.append(zip(atom_types, positions))
                    residues.append(j)
                    polarities.append(polarity)
            expected_peaks[sname] = peaks
            self.e_polarities[sname] = polarities
            self.residues[sname] = np.array(residues)

        # define set of assignable atoms and return dictionary to redefine
        # expected peaks
        atom_dict = self.get_assignable_atoms(expected_peaks)

        self.expected_peaks = {}
        for sname, peaks in expected_peaks.iteritems():
            peaks = [tuple(atom_dict[a] for a in peak) for peak in peaks]
            self.expected_peaks[sname] = peaks

    def set_spin_systems(self):

        """
        Sets clean set of spin systems.
        """

        residues = self.protein.residues
        self.spin_systems = np.ones((residues.shape[-1], 6)) * np.nan
        self.spin_systems[:, [0, 1, 2, 4]] = residues.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.spin_systems[1:, [3, 5]] = residues.loc[['CA', 'CB']].as_matrix()[:, :-1].T
        n = self.spin_systems.shape[0]
        self.spin_systems[1:, 3] += np.random.randn(n-1) * IPASS_SETTINGS['std']['CA']
        self.spin_systems[1:, 5] += np.random.randn(n-1) * IPASS_SETTINGS['std']['CB']

        mask = np.sum(np.isnan(self.spin_systems[:, [0, 1]]), axis=1) == 0
        self.spin_to_residue = np.arange(n, dtype=np.int32)
        self.spin_to_residue = self.spin_to_residue[mask]
        self.spin_systems = self.spin_systems[mask, :]

        self.true_spin_systems = np.ones((residues.shape[-1], 6)) * np.nan
        self.true_spin_systems[:, [0, 1, 2, 4]] = residues.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.true_spin_systems[1:, [3, 5]] = residues.loc[['CA', 'CB']].as_matrix()[:, :-1].T

        # means and stds of prior
        locs = self.protein.locs
        self.spin_locs = np.ones((locs.shape[-1], 6)) * np.nan
        self.spin_locs[:, [0, 1, 2, 4]] = locs.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.spin_locs[1:, [3, 5]] = locs.loc[['CA', 'CB']].as_matrix()[:, :-1].T

        scales = self.protein.scales
        self.spin_scales = np.ones((scales.shape[-1], 6)) * np.nan
        self.spin_scales[:, [0, 1, 2, 4]] = scales.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.spin_scales[1:, [3, 5]] = scales.loc[['CA', 'CB']].as_matrix()[:, :-1].T


    def get_assignable_atoms(self, expected_peaks):

        """
        Get a list of assignable atoms and modify
        """
        atoms = set()

        for sname, peaks in expected_peaks.iteritems():
            for peak in peaks:
                for atom in peak:
                    atoms.add(atom)

        self.atoms = list(atoms)

        return {a: i for i, a in enumerate(self.atoms)}

    def get_residue_list(self):

        """
        Sets the list of residues in terms of the assignable atoms
        given the experiments.
        """

        sequence = self.protein.sequence
        nr = len(sequence)

        atom_types = list(set([a[0] for a in self.atoms]))

        residues = pd.DataFrame(data=np.ones((len(atom_types), nr))*-1,
                                index=atom_types,
                                dtype=np.int32)

        for n, a in enumerate(self.atoms):
            i, j = a
            residues.loc[i, j] = n

        return residues

    @property
    def residue_array(self):

        """
        Return numpy masked array for indexes into atoms for each of the
        protein's residues.
        """

        r = self.residues.as_matrix()
        mr = np.ma.masked_array(r, r == -1)

        return mr


class SpinSystemExperiment(object):

    """
    A simulated experiment including measured peaks, expected peaks, and
    a set of assignable atoms
    """

    def __init__(self, protein, spectra, spin_file):

        self.protein = protein
        self.spectra = spectra
        self.set_expected_peaks()   # also initializes self.atoms
        self.set_spin_systems(spin_file)
        self.true_x = np.array(
            [self.protein.atom_frequency(a) for a in self.atoms])

    def set_expected_peaks(self):

        """
        Get expected set of peaks.
        """
        expected_peaks = {}
        self.e_polarities = {}
        self.residues = {}

        for sname in self.spectra:
            peaks = []
            polarities = []
            residues = []
            spectrum = SPECTRA[sname]
            for peak, polarity in zip(spectrum['peaks'],
                                      spectrum['polarities']):
                for j in xrange(self.protein.size):
                    atom_types, positions = zip(*peak)
                    positions = np.array(positions) + j
                    if np.any(positions < 0):
                        continue
                    aas = [self.protein.sequence[pos] for
                           pos in positions]
                    mus = [DATA['mean'].loc[aa, atype]
                           for aa, atype in zip(aas, atom_types)]
                    # check if data exists for this aminoacid
                    if np.any(np.isnan(mus)):
                        continue
                    peaks.append(zip(atom_types, positions))
                    residues.append(j)
                    polarities.append(polarity)
            expected_peaks[sname] = peaks
            self.e_polarities[sname] = polarities
            self.residues[sname] = np.array(residues)

        # define set of assignable atoms and return dictionary to redefine
        # expected peaks
        atom_dict = self.get_assignable_atoms(expected_peaks)

        self.expected_peaks = {}
        for sname, peaks in expected_peaks.iteritems():
            peaks = [tuple(atom_dict[a] for a in peak) for peak in peaks]
            self.expected_peaks[sname] = peaks

    def get_assignable_atoms(self, expected_peaks):

        """
        Get a list of assignable atoms and modify
        """
        atoms = set()

        for sname, peaks in expected_peaks.iteritems():
            for peak in peaks:
                for atom in peak:
                    atoms.add(atom)

        self.atoms = list(atoms)

        return {a: i for i, a in enumerate(self.atoms)}

    def get_residue_list(self):

        """
        Sets the list of residues in terms of the assignable atoms
        given the experiments.
        """

        sequence = self.protein.sequence
        nr = len(sequence)

        atom_types = list(set([a[0] for a in self.atoms]))

        residues = pd.DataFrame(data=np.ones((len(atom_types), nr))*-1,
                                index=atom_types,
                                dtype=np.int32)

        for n, a in enumerate(self.atoms):
            i, j = a
            residues.loc[i, j] = n

        return residues

    def set_spin_systems(self, spin_file):

        """
        Loads spin systems from file.
        """

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
                spins.append(current)

        self.spin_systems = np.array(spins)

        residues = self.protein.residues
        self.true_spin_systems = np.ones((residues.shape[-1], 6)) * np.nan
        self.true_spin_systems[:, [0, 1, 2, 4]] = residues.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.true_spin_systems[1:, [3, 5]] = residues.loc[['CA', 'CB']].as_matrix()[:, :-1].T

        self.q = np.array([IPASS_SETTINGS['std']['N'],
                           IPASS_SETTINGS['std']['HN'],
                           IPASS_SETTINGS['std']['CA'],
                           IPASS_SETTINGS['std']['CA'],
                           IPASS_SETTINGS['std']['CB'],
                           IPASS_SETTINGS['std']['CB']])
        self.tol = np.array([IPASS_SETTINGS['tolerance']['N'],
                             IPASS_SETTINGS['tolerance']['HN'],
                             IPASS_SETTINGS['tolerance']['CA'],
                             IPASS_SETTINGS['tolerance']['CA'],
                             IPASS_SETTINGS['tolerance']['CB'],
                             IPASS_SETTINGS['tolerance']['CB']])

        # determine true matches
        self.true_matches = [self.match(s) for s in self.spin_systems]
        self.residue_to_spin = [self.match(s, target='spins')
                                for s in self.true_spin_systems]

        # means and stds of prior
        locs = self.protein.locs
        self.spin_locs = np.ones((locs.shape[-1], 6)) * np.nan
        self.spin_locs[:, [0, 1, 2, 4]] = locs.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.spin_locs[1:, [3, 5]] = locs.loc[['CA', 'CB']].as_matrix()[:, :-1].T

        scales = self.protein.scales
        self.spin_scales = np.ones((scales.shape[-1], 6)) * np.nan
        self.spin_scales[:, [0, 1, 2, 4]] = scales.loc[['N', 'H', 'CA', 'CB'], :].as_matrix().T
        self.spin_scales[1:, [3, 5]] = scales.loc[['CA', 'CB']].as_matrix()[:, :-1].T

    def __distance(self, s1, target):

        """
        Distance between spin system s1 and each true spin
        """

        z = np.zeros(target.shape[0]) * np.nan
        for i in range(target.shape[0]):
            s2 = target[i, :]
            mask = np.logical_and(~np.isnan(s1), ~np.isnan(s2))
            if np.sum(mask) < 0.5:
                continue
            diffs = np.abs((s1[mask]-s2[mask]))/self.q[mask]
            # option 1 - mean amount of standard deviations away
            z[i] = np.mean(diffs)
            # option 2 - maximum amount of standard deviations away
            # z[i] = np.max(diffs)
        return z

    def match(self, s1, threshold=2.5, target='true_spins'):

        if target == 'true_spins':
            target = self.true_spin_systems
        elif target == 'spins':
            target = self.spin_systems
        else:
            raise ValueError("Target must be 'true_spins' or 'spins'.")

        z = self.__distance(s1, target)
        check = np.nanmin(z)
        if check < threshold:
            # print 'match found: {} to {}'.format(s1, self.true_spin_systems[np.nanargmin(z), :])
            return np.nanargmin(z)
        return None

    @property
    def residue_array(self):

        """
        Return numpy masked array for indexes into atoms for each of the
        protein's residues.
        """

        r = self.residues.as_matrix()
        mr = np.ma.masked_array(r, r == -1)

        return mr
