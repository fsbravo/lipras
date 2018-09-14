from itertools import count
import copy

import numpy as np


######################################
# EXPECTED PEAKS #####################
######################################


class Peak(object):

    """
    Expected peak
    """

    _ids = count(0)

    def __init__(self, spectrum, indices, residue, polarity, sequence):

        self.residue = residue
        self.residue_type = sequence[residue]
        self.spectrum = spectrum
        self.polarity = polarity
        self.id = next(self._ids)
        self.__indices__ = np.array(indices, dtype=np.int32)

    @property
    def value(self):

        return self.__indices__

    def __repr__(self):

        return 'Peak({}, residue={} ({}), polarity={})'.format(
            self.value, self.residue, self.residue_type,
            self.polarity)

    def __repr__(self):

        return 'Peak({}, residue={} ({}), polarity={})'.format(
            self.value, self.residue, self.residue_type,
            self.polarity)

    def contains(self, atomid):

        return atomid in self.__indices__

    def raw_lower(self, atoms):

        return np.array([atoms[i].raw_lower for i in self.value])

    def raw_upper(self, atoms):

        return np.array([atoms[i].raw_upper for i in self.value])


class Spectrum(object):

    """
    Spectrum (collection of expected peaks)
    """

    def __init__(self, name, data, peaks=None):

        self.name = name
        if peaks is not None:
            self.peaks = peaks
        else:
            peaks = data['peaks'][name]
            residues = data['residues'][name]
            polarity = data['polarities'][name]
            sequence = data['sequence']
            self.peaks = [Peak(name, p, r, pl, sequence) for
                          p, r, pl in zip(peaks, residues, polarity)]
        self.__value__ = np.array([p.value for p in self.peaks], dtype=np.int32)
        self.__polarities__ = np.array([p.polarity for p in self.peaks])
        self.__residues__ = np.array([p.residue for p in self.peaks])
        self.n = self.__value__.shape[0]

    @property
    def value(self):

        return self.__value__

    @property
    def polarities(self):

        return self.__polarities__

    @property
    def residues(self):

        return self.__residues__

    def __repr__(self):

        return 'Spectrum({}: {} peaks)'.format(self.name, self.n)

    def __str__(self):

        return 'Spectrum({}: {} peaks)'.format(self.name, self.n)

    def __getitem__(self, i):

        return self.peaks[i]

    def describe(self):

        print 'Spectrum({}:\n\t'.format(self.name) + \
            '\n\t'.join(str(p) for p in self.peaks) + ')'

    def count(self, atomid):

        return np.sum(self.__value__ == atomid)

    def find(self, atomid):

        """
        Find observations of atom <atomid> in peaks.
        """

        return [p for p in self.peaks if p.contains(atomid)]

    def subset_to(self, i):

        peaks = [p for p in self.peaks if p.residue == i]
        return Spectrum(self.name, None, peaks=peaks)


class ExpectedPeaks(dict):

    """
    Set of expected peaks.
    """

    def __init__(self, data=None, spectra=None):

        assert data is not None or spectra is not None, \
            'one of data or spectra must be provided'

        if data is not None:
            for spectrum in data['peaks'].keys():
                self[spectrum] = Spectrum(
                    spectrum, data)
        else:
            for spectrum in spectra:
                self[spectrum.name] = spectrum

        self.atom_list = list(
            set(np.concatenate(
                [s.value.reshape(-1) for s in self.values()])))

    def __repr__(self):

        return 'ExpectedPeaks(\n\t{}\n\t)'.format(
            '\n\t'.join([str(spectrum) for spectrum in self.values()]))

    def __repr__(self):

        return 'ExpectedPeaks(\n\t{}\n\t)'.format(
            '\n\t'.join([str(spectrum) for spectrum in self.values()]))

    @property
    def value(self):

        return [(s.name, s.value) for s in self.values()]

    def describe(self):

        for v in self.values():
            v.describe()

    def subset_to(self, i):

        """
        Returns ExpectedPeaks object for peaks in residue i.
        """

        spectra = [s.subset_to(i) for s in self.values()]
        return ExpectedPeaks(spectra=spectra)

    def count(self, atomid):

        """
        Counts number of observations of atom.
        """

        return np.sum([s.count(atomid) for s in self.values()])

    def find(self, atomid):

        """
        Finds observations of atom <atomid> in peaks.
        """

        tmp = []
        for spectrum in self.values():
            tmp += spectrum.find(atomid)
        return tmp

    def polarity(self, key):

        return self[key].polarities

    def __getitem__(self, i):

        if isinstance(i, tuple):
            return super(ExpectedPeaks, self).__getitem__(i[0])[i[1]]
        else:
            return super(ExpectedPeaks, self).__getitem__(i)
