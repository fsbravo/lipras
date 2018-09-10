import networkx as nx
import numpy as np
from common import *

from scipy.spatial.distance import cdist
from itertools import count
from collections import defaultdict, OrderedDict


######################################
# MEASURED PEAKS #####################
######################################

class MeasuredPeak(object):

    """
    Measured peak
    """

    _ids = count(0)

    def __init__(self, spectrum, values, polarity, u):

        self.spectrum = spectrum
        self.polarity = polarity
        self.__u__ = u
        self.__values__ = values
        self.tolerance = np.array([PARAMS['tolerance']['N'],
                                   PARAMS['tolerance']['H'],
                                   PARAMS['tolerance']['C']])
        self.__id__ = next(self._ids)
        self.__dim__ = len(values)

    @property
    def dim(self):

        return self.__dim__

    @property
    def id(self):

        return self.__id__

    def __hash__(self):

        return self.id

    @property
    def value(self):

        return self.__values__

    @property
    def u(self):

        return self.__u__

    def __repr__(self):

        return 'MeasuredPeak({}, {})'.format(
            self.value, self.spectrum)

    def __repr__(self):

        return 'MeasuredPeak({}, {})'.format(
            self.value, self.spectrum)

    def is_consistent_with(self, peak):

        """
        Used to verify consistency.

        The factor of 2 comes accounts for the possibility that the peak
        is at the edge of the bounding box of the atom.
        """

        l = self.value[:2] - self.tolerance[:2]
        u = self.value[:2] + self.tolerance[:2]
        return not (np.any(l > peak.value[:2]) or np.any(u < peak.value[:2]))

    def is_C_consistent_with(self, peak):

        """
        Asserts consistency for carbon atoms.
        """

        if self.dim < 3 or peak.dim < 3:
            return False

        l = self.value[2] - self.tolerance[2]
        u = self.value[2] + self.tolerance[2]
        return peak.value[2] < u and peak.value[2] > l


class MeasuredSpectrum(object):

    """
    Spectrum (collection of measured peaks)
    """

    def __init__(self, name, experiment, peaks=None):

        self.name = name
        if peaks is not None:
            self.peaks = peaks
        else:
            peaks = experiment.measured_peaks[name]
            polarity = experiment.m_polarities[name]
            us = np.eye(peaks.shape[0])
            self.peaks = [MeasuredPeak(name, p, pl, u) for
                          p, pl, u in zip(peaks, polarity, us)]
        self.__u__ = np.array([p.u for p in self.peaks])
        self.__value__ = np.array([p.value for p in self.peaks])
        self.__polarities__ = np.array([p.polarity for p in self.peaks])

        self.n = self.__value__.shape[0]

    @property
    def value(self):

        return self.__value__

    @property
    def u(self):

        return self.__u__

    @property
    def polarities(self):

        return self.__polarities__

    def __len__(self):

        return len(self.peaks)

    def __repr__(self):

        return 'MeasuredSpectrum({}: {} peaks)'.format(self.name, self.n)

    def __str__(self):

        return 'MeasuredSpectrum({}: {} peaks)'.format(self.name, self.n)

    def __getitem__(self, i):

        return self.peaks[i]

    def describe(self):

        print 'MeasuredSpectrum({}:\n\t'.format(self.name) + \
            '\n\t'.join(str(p) for p in self.peaks) + ')'

    def consistent_with(self, peak, as_matrix=False):

        """
        Returns MeasuredSpectrum object of peaks that are consistent
        with given peak.

        If as_matrix is True returns only values instead (faster).
        """

        nh_tolerance = np.array([PARAMS['tolerance']['N'],
                                 PARAMS['tolerance']['H']])

        lower = peak[:2] - nh_tolerance
        upper = peak[:2] + nh_tolerance

        return self.peaks_in_window(lower, upper, as_matrix)

    def peaks_in_window(self, lower, upper, as_matrix=False):

        """
        Returns peaks in the given window.

        By default returns MeasuredSpectrum object. If as_matrix is set to True
        then returns only values instead (faster).
        """

        mask = np.logical_and(np.all(self.value[:, :2] > lower.reshape((1, -1)), axis=1),
                              np.all(self.value[:, :2] < upper.reshape((1, -1)), axis=1))

        if as_matrix:
            return self.value[mask, :]

        peaks = [p for check, p in zip(mask, self.peaks) if check]
        return MeasuredSpectrum(self.name, None, peaks=peaks)

    def subset_to(self, window):

        """
        Subsets to peaks consistent with given Window object.
        """

        mask = window.is_in(self.value)
        peaks = [p for check, p in zip(mask, self.peaks) if check]

        return MeasuredSpectrum(self.name, None, peaks=peaks)

    def sorted(self, residue):

        """
        Returns MeasuredPeaks object sorted by the normalized distance
        to the NH pair of the residue.
        """

        N = residue.N
        H = residue.H
        if N is None or H is None:
            return MeasuredSpectrum(self.name, None, peaks=self.peaks)

        base = np.array([N.mu, H.mu])
        std = np.array([N.std, H.std])
        Y = np.reshape(base / std, (1, -1))

        X = self.value[:, :2] / np.reshape(std, (1, -1))
        distances = np.reshape(cdist(X, Y), (-1, ))
        order = np.argsort(distances)

        peaks = [self.peaks[i] for i in order]
        return MeasuredSpectrum(self.name, None, peaks=peaks)


class MeasuredPeaks(dict):

    """
    Set of measured peaks.
    """

    _ids = count(0)

    def __init__(self, experiment=None, spectra=None, ushape=None, *varargs, **kwargs):

        assert experiment is not None or spectra is not None, \
            'one of experiment or spectra must be provided'

        if experiment is not None:
            for spectrum in experiment.measured_peaks.keys():
                self[spectrum] = MeasuredSpectrum(
                    spectrum, experiment)
            self.__ushape__ = OrderedDict(
                [(key, self[key].u.shape[-1]) for key in self.keys()])
            # self.__ushape__ = {key: self[key].u.shape[-1] for key in self.keys()}
        else:
            for spectrum in spectra:
                self[spectrum.name] = spectrum
            self.__ushape__ = ushape

        self.id = next(self._ids)

        super(MeasuredPeaks, self).__init__(*varargs, **kwargs)

    @property
    def ushape(self):

        return self.__ushape__

    def __repr__(self):

        return 'MeasuredPeaks(\n\t{}\n\t)'.format(
            '\n\t'.join([str(spectrum) for spectrum in self.values()]))

    def __repr__(self):

        return 'MeasuredPeaks(\n\t{}\n\t)'.format(
            '\n\t'.join([str(spectrum) for spectrum in self.values()]))

    def __getitem__(self, i):

        if isinstance(i, tuple):
            return super(MeasuredPeaks, self).__getitem__(i[0])[i[1]]
        else:
            return super(MeasuredPeaks, self).__getitem__(i)

    def describe(self):

        print 'MeasuredPeaks('
        for spectrum in self.values():
            spectrum.describe()
        print ')'

    def polarity(self, key):

        return self[key].polarities

    def split_on(self, key):

        """
        Splits into multiple MeasuredPeaks object based on NH consistency.
        """

        assert key in self.keys(), 'ERROR: spectrum {} not available!'.format(
            key)

        splits = []
        for p in self[key].peaks:
            spectra = [self[key].consistent_with(p) for key in self.keys()]
            splits.append(MeasuredPeaks(spectra=spectra, ushape=self.__ushape__))

        return splits

    def subset_to(self, window):

        """
        Returns MeasuredPeaks object including only peaks consistent with
        given Window object.
        """

        spectra = [self[key].subset_to(window) for key in self.keys()]

        return MeasuredPeaks(spectra=spectra, ushape=self.__ushape__)

    def sorted(self, residue):

        """
        Sorts peaks by the normalized distance to the NH pair of the residue.
        """

        spectra = [self[key].sorted(residue) for key in self.keys()]

        return MeasuredPeaks(spectra=spectra, ushape=self.__ushape__)


class Measured(MeasuredPeaks):

    """
    Measured peaks with graph functionalities.
    """

    def __init__(self, experiment=None, spectra=None, ushape=None, peaks=None):

        super(Measured, self).__init__(experiment, spectra, ushape)

        self.G = nx.Graph()
        self.G_C = nx.Graph()

        self._create_nodes()
        self._create_edges()

    def _create_nodes(self):

        for spectrum in self.values():
            self.G.add_nodes_from((p for p in spectrum))
            self.G_C.add_nodes_from((p for p in spectrum))

    def _create_edges(self):

        for node1 in self.G.nodes():
            self.G.add_edges_from(
                (node1, node2) for node2 in self.G.nodes()
                if node1.is_consistent_with(node2))
            self.G_C.add_edges_from(
                (node1, node2) for node2 in self.G.nodes()
                if node1.is_C_consistent_with(node2))

    def _clique(self, clique):

        """
        Transforms a clique into a Measured object.
        """

        peaks = defaultdict(list)
        for peak in clique:
            peaks[peak.spectrum].append(peak)
        spectra = [MeasuredSpectrum(key, None, peaks=value)
                   for key, value in peaks.iteritems()]
        return Measured(spectra=spectra, ushape=self.__ushape__)

    def clique_generator(self, spectrum):

        """
        Returns all cliques (based on peaks in given spectrum).
        """

        peaks = self[spectrum].peaks
        order = np.random.permutation(len(peaks))
        for i in order:
            peak = peaks[i]
            cliques = nx.cliques_containing_node(self.G, peak)
            for clique in cliques:
                yield self._clique(clique)

    def all_cliques(self, spectrum):

        """
        Returns all cliques (based on peaks in given spectrum).
        """

        peaks = self[spectrum].peaks
        cliques = []
        for peak in peaks:
            current = nx.cliques_containing_node(self.G, peak)
            cliques += [self._clique(clique) for clique in current]
        return cliques

    def subset_to(self, window):

        """
        Returns MeasuredGraph object consistent with window.
        """

        spectra = [self[key].subset_to(window) for key in self.keys()]

        return Measured(spectra=spectra, ushape=self.__ushape__)

    def sorted(self, residue, spectrum=None):

        """
        Returns MeasureGraph with peaks sorted by distance to residue base pair.
        """

        if spectrum is not None:
            spectra = [self[key].sorted(residue) if key == spectrum else self[key]
                       for key in self.keys()]
        else:
            spectra = [self[key].sorted(residue) for key in self.keys()]

        return Measured(spectra=spectra, ushape=self.__ushape__)


class Pairing(object):

    def __init__(self, spectrum, epeak, mpeak, level=None):

        self.spectrum = spectrum
        self.epeak = epeak
        self.mpeak = mpeak
        self.level = level

    def __hash__(self):

        return hash((self.spectrum, self.epeak, self.mpeak))


class Configuration(dict):

    def __init__(self, ushape, clique=[]):

        super(Configuration, self).__init__()

        u = OrderedDict([(key, np.zeros(value))
                         for key, value in ushape.iteritems()])

        for pairing in clique:
            for atom, value in zip(pairing.epeak, pairing.mpeak.value):
                try:
                    self[atom].append(value)
                except KeyError:
                    self[atom] = [value]
            u[pairing.spectrum] += pairing.mpeak.u

        self.__u__ = np.concatenate(u.values())

    @property
    def u(self):

        return self.__u__


class MeasuredConfigurator(object):

    """
    A configuration for a Measured NH clique.

    Forms the backbone for a generator of configurations of measured peaks
    that is consistent with the chosen spectra.

    Produces different configurations of measured peaks that match the
    expected peak structure (defined by the chosen spectra).

    If complete flag is set to True then produces all valid configurations
    (this greatly increases the number of configurations).

    If complete flag is set to False then SCHEDULE (as defined in common.py)
    is used to determine the assignment order, and only configurations
    produced by that schedule are accepted.
    """

    def __init__(self, spectra, m_clique, complete=True):

        super(MeasuredConfigurator, self).__init__()

        self.G = m_clique.G_C

        # setup schedule
        self.schedule = OrderedDict()
        for spectrum, peak_no in SCHEDULE:
            if spectrum not in spectra:
                continue
            epeak = SPECTRA[spectrum]['peaks'][peak_no]
            polarity = SPECTRA[spectrum]['polarities'][peak_no]
            valid = (m_peak for m_peak in self.G.nodes() if
                     m_peak.spectrum == spectrum and m_peak.polarity == polarity)
            self.schedule[spectrum, epeak] = list(valid)

        # setup exploration graph
        self.E = nx.Graph()

        for (spectrum, epeak), options in self.schedule.iteritems():
            for mpeak in options:
                self.E.add_node(Pairing(spectrum, epeak, mpeak))

        for node_1 in self.E.nodes():
            for node_2 in self.E.nodes():
                if node_1.mpeak == node_2.mpeak:
                    continue
                if node_1.epeak != node_2.epeak:
                    self.E.add_edge(node_1, node_2)
                elif node_1.mpeak in self.G[node_2.mpeak]:
                    self.E.add_edge(node_1, node_2)

        cliques = sorted(list(nx.find_cliques(self.E)),
                         key=lambda c: len(c), reverse=True)

        ushape = m_clique.ushape
        self.__configurations__ = [Configuration(ushape, clique) for clique in cliques]

    @property
    def configurations(self):

        return self.__configurations__

    def __getitem__(self, i):

        return self.__configurations__[i]
