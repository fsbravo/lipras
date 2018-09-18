from BaseStructures import *
from common import *
from Measured import *

from matplotlib import pyplot as plt
import tqdm

import scipy.stats as sps
import numpy as np
from cvxpy import *

import networkx as nx

from collections import defaultdict, OrderedDict
from itertools import count
import copy

from colorama import Fore, Back, Style


######################################
# ATOMS ##############################
######################################

def score(values, mu, sigma, q):

    """
    Computes integral over x of f(pi|x)f(x) for Gaussian f

    arguments
        :values:    values assigned to atom
        :mu:        atom mean (prior)
        :sigma:     atom variance (prior)
        :q:         atom variance (noise)
    """

    d = float(len(values))
    values = np.array(values)
    sigmabar = 1. / (d/q + 1./sigma)
    mubar = sigmabar * (np.sum(values)/q + mu/sigma)
    return -d / 2. * np.log(2. * np.pi) + \
        0.5 * (np.log(sigmabar) - np.log(sigma) - d*np.log(q)) - \
        0.5 * (np.sum(values ** 2)/q + mu ** 2/sigma - mubar**2/sigmabar)

# pre-compute thresholds
threshold = PARAMS['fa_threshold']
THRESHOLDS = {}
for atm_type in PARAMS['tolerance'].keys():
    tol = PARAMS['tolerance'][atm_type]
    q = PARAMS['std'][atm_type]
    half_tol_threshold = PARAMS['half_tol_threshold']*tol
    for res_type in DATA['stddev'].index:
        stddev = DATA['stddev'].loc[res_type, atm_type]
        for i in range(20):
            values = [threshold*stddev - 2.*half_tol_threshold*(i % 2 == 0)
                      for _ in range(i)]
            THRESHOLDS[(atm_type, res_type, i)] = score(values, 0, stddev**2, q**2)


class AtomFA(Atom):

    def __init__(self, atom_type, residue, atom_id, sequence, peaks):

        super(AtomFA, self).__init__()

        self.__residue__ = residue
        self.__atom_type__ = atom_type
        residue_type = sequence[residue]
        self.__residue_type__ = residue_type

        self.__mu__ = DATA['mean'].loc[residue_type, atom_type]
        self.__std__ = DATA['stddev'].loc[residue_type, atom_type]
        self.__var__ = self.std ** 2
        self.__qstd__ = PARAMS['std'][atom_type]
        self.__qvar__ = self.qstd ** 2
        self.__tolerance__ = PARAMS['tolerance'][atom_type]

        self.__id__ = atom_id

        # custom
        self.__value__ = np.nan
        self.__values__ = []

        # peaks where atom is observed
        self.observations = peaks.observations[atom_id]
        # self.observations = peaks.find(atom_id)
        # residues where atom appears
        self.residues = set([o.residue for o in self.observations])
        self.total_obs = len(self.observations)
        self.residue_obs = len(
            [p for p in self.observations if p.residue == residue])

        # thresholds
        threshold = PARAMS['fa_threshold']
        half_tol_threshold = self.tolerance * PARAMS['half_tol_threshold']

        values = [self.mu + threshold*self.std - 2.*half_tol_threshold*(i % 2 == 0)
                  for i in range(self.residue_obs)]
        self.__node_threshold__ = THRESHOLDS[self.atom_type, self.residue_type, self.residue_obs]
        # self.__node_threshold__ = score(values, self.mu, self.var, self.qvar)

        values = [self.mu + threshold*self.std - 2.*half_tol_threshold*(i % 2 == 0)
                  for i in range(self.total_obs)]
        self.__node_threshold__ = THRESHOLDS[self.atom_type, self.residue_type, self.total_obs]
        # self.__edge_threshold__ = score(values, self.mu, self.var, self.qvar)

    def __str__(self):

        return 'Atom({}{} ({}))'.format(self.residue,
                                        self.atom_type,
                                        self.residue_type)

    def __repr__(self):

        return 'Atom({}{} ({}))'.format(self.residue,
                                        self.atom_type,
                                        self.residue_type)

    @property
    def value(self):

        return self.__value__

    @property
    def values(self):

        return self.__values__

    @property
    def assigned(self):

        return len(self.__values__) > 0

    def assign(self, v):

        if isinstance(v, float):
            self.__values__.append(v)
        else:
            self.__values__ += list(v)
        self.__value__ = np.mean(self.__values__)

    def clear(self):

        self.__values__ = []
        self.__value__ = np.nan

    def merge(self, other):

        atom = self.copy()
        atom.assign(other.__values__)

        return atom

    def node_score(self):

        if not self.assigned:
            return self.__node_threshold__

        values = self.__values__
        if len(values) < self.residue_obs:
            half_tol_threshold = self.tolerance * PARAMS['half_tol_threshold']
            # values += [self.value + (((i % 2 == 0) * 2) - 1) * self.tolerance
            #            for i in range(self.residue_obs - len(values))]
            values += [self.value + ((i % 2 == 0) - 0.5) * half_tol_threshold
                       for i in range(self.residue_obs - len(values))]

        return score(values, self.mu, self.var, self.qvar)

    def edge_score(self, other):

        if other is None:
            return self.node_score()

        if not self.assigned and not other.assigned:
            return self.__edge_threshold__

        values = self.__values__ + other.__values__
        if len(values) < self.total_obs:
            value = np.mean(values)
            half_tol_threshold = self.tolerance * PARAMS['half_tol_threshold']
            values += [value + ((i % 2 == 0) - 0.5) * half_tol_threshold
                       for i in range(self.total_obs - len(values))]

        return score(values, self.mu, self.var, self.qvar)


######################################
# ATOM GROUP #########################
######################################

class AtomGroupFA(AtomGroup):

    """
    A group of atoms (typically atoms corresponding to one observable spin
    system or to a contiguous fragment).

    These can be added together to form longer fragments.
    """

    def __init__(self, sequence, atoms, group_residue, peaks,
                 *varargs, **kwargs):

        """
        residue = first residue in the fragment
        """

        super(AtomGroupFA, self).__init__(*varargs, **kwargs)

        for atom_type, residue, atom_id in atoms:

            self[atom_id] = AtomFA(atom_type, residue, atom_id, sequence, peaks)
            self.__idkeys__[atom_type, residue] = atom_id

        self.residue = group_residue

        # compute thresholds
        self.__node_threshold__ = np.sum(
            [a.node_threshold for a in self.values()])
        self.__edge_threshold__ = np.sum(
            [a.edge_threshold for a in self.values()])

    def node_score(self):

        return np.sum([a.node_score() for a in self.values()])

    def edge_score(self, other):

        common = self.intersection(other)
        unique = set(self.keys()) - set(common)
        score = np.sum([self[atom_no].edge_score(other[atom_no])
                        for atom_no in common])
        score += np.sum([self[atom_no].edge_score(None) for atom_no in unique])

        return score

    def assign_configuration(self, configuration):

        """
        Assign values from configurations.
        """

        for (atom_type, shift), values in configuration.iteritems():
            atom_no = self.residue + shift
            try:
                self[atom_type, atom_no].assign(values)
            except KeyError:
                continue

        self.__u__ = configuration.u


######################################
# RESIDUE ############################
######################################

class Residue(AtomGroupFA):

    def __init__(self, sequence, atoms, group_residue, peaks):

        super(Residue, self).__init__(sequence, atoms, group_residue, peaks)
        self.sequence = sequence
        self.atoms = atoms
        self.group_residue = group_residue
        self.peaks = peaks
        self.local_peaks = peaks.subset_to(group_residue)

    # generator for nodes
    def samples(self, configurator, n=None):

        if n is None:
            n = len(configurator.configurations)
        for configuration in configurator.configurations[:n+1]:
            group = AtomGroupFA(self.sequence, self.atoms, self.group_residue, self.peaks)
            group.assign_configuration(configuration)
            yield group

    def empty_sample(self, configuration):

        group = AtomGroupFA(self.sequence, self.atoms, self.group_residue, self.peaks)
        group.assign_configuration(configuration)
        return group


######################################
# ASSIGNER ###########################
######################################

class AssignerFA(Assigner):

    def __init__(self, experiment, *varargs, **kwargs):

        super(AssignerFA, self).__init__(experiment, *varargs, **kwargs)

    def _setup(self):

        self.peaks = self.experiment.expected
        self.measured = self.experiment.measured
        self.residues = []
        atoms = [(t[0], t[1], i) for i, t in enumerate(self.experiment.atoms)]
        for i in range(self.n):
            atom_list = self.peaks.subset_to(i).atom_list
            atom_tuples = [atoms[j] for j in atom_list]
            self.residues.append(Residue(self.sequence,
                                         atom_tuples,
                                         group_residue=i,
                                         peaks=self.peaks))

    def _create_nodes(self, verbose=False):

        print 'creating nodes ...'

        cliques = self.measured.all_cliques('HSQC')
        self.layers = [[] for _ in range(self.n)]
        for i, clique in tqdm.tqdm(enumerate(cliques), total=len(cliques)):
            # print 'clique %d' % i
            configurator = MeasuredConfigurator(self.peaks.keys(), clique)
            for residue in self.residues:
                for sample in residue.samples(configurator, n=2):
                    if sample.node_score() > sample.node_threshold:
                        self.add(sample)

        # add empty nodes at the end
        empty_configuration = Configuration(self.measured.ushape)
        for residue in self.residues:
            sample = residue.empty_sample(empty_configuration)
            self.add(sample)

        # take arbitrary configurator
        configurator = MeasuredConfigurator(self.peaks.keys(), cliques[0])
        self.start = AtomGroupFA(
            self.sequence, [], group_residue=None, peaks=self.peaks)
        self.start.assign_configuration(empty_configuration)
        self.add(self.start)
        self.end = AtomGroupFA(
            self.sequence, [], group_residue=None, peaks=self.peaks)
        self.end.assign_configuration(empty_configuration)
        self.add(self.end)

        print '... created %d nodes' % len(self.nodes())

    def _create_edges(self, verbose=False):

        min_weight = np.inf

        print 'creating edges ...'

        for i in tqdm.tqdm(range(self.n-1), total=self.n-1):
            # print '   ...layer {} -> {}'.format(i, i+1)
            layer = self.layers[i]
            for node in layer:
                o_layer = self.layers[node.residue+1]
                for o_node in o_layer:
                    score = node.edge_score(o_node)
                    if score < node.edge_threshold - 1.:
                        continue
                    self.add_edge(node, o_node)
                    # adjust score for relaxation
                    self[node][o_node]['score'] = node.edge_score(o_node)
                    # TEST
                    self[node][o_node]['relaxed'] = score - 100. * np.sum(o_node.u)
                    self[node][o_node]['weight'] = -self[node][o_node]['relaxed']
                    # END TEST
                    # self[node][o_node]['weight'] = -self[node][o_node]['score']
                    # END BACKUP
                    if self[node][o_node]['weight'] < min_weight:
                        min_weight = self[node][o_node]['weight']

        # start and end edges
        for node in self.layers[0]:
            self.add_edge(self.start, node)
            self[self.start][node]['score'] = 0.
            self[self.start][node]['weight'] = 1.e-6

        for node in self.layers[-1]:
            self.add_edge(node, self.end)
            self[node][self.end]['score'] = 0.
            self[node][self.end]['weight'] = 1.e-6

        if min_weight < 1.:
            for edge in self.edges():
                self[edge[0]][edge[1]]['weight'] -= (min_weight - 1.)

        print '... created %d edges' % len(self.edges())
