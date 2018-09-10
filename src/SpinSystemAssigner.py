from BaseStructures import *
from common import *

from matplotlib import pyplot as plt
import tqdm

import scipy.stats as sps
import numpy as np
from scipy.optimize import linear_sum_assignment
from cvxpy import *

import networkx as nx

from collections import defaultdict, OrderedDict
from itertools import count
import copy

from colorama import Fore, Back, Style


######################################
# ATOMS ##############################
######################################

class AtomSS(Atom):

    def __init__(self, atom_type, residue, atom_id, sequence, gamma=1.):

        super(AtomSS, self).__init__()

        self.__residue__ = residue
        self.__atom_type__ = atom_type
        residue_type = sequence[residue]
        self.__residue_type__ = residue_type

        self.__mu__ = DATA['mean'].loc[residue_type, atom_type]
        self.__std__ = DATA['stddev'].loc[residue_type, atom_type]
        self.__var__ = self.std ** 2
        # self.__qstd__ = IPASS_SETTINGS['std'][atom_type]
        self.__qstd__ = IPASS_SETTINGS['q'][atom_type]
        self.__qvar__ = self.qstd ** 2
        self.__tolerance__ = IPASS_SETTINGS['tolerance'][atom_type]

        self.__id__ = atom_id

        threshold = sps.norm.logpdf(PARAMS['threshold'])
        self.__node_threshold__ = threshold
        # TODO: fix edge threshold (if next residue does not have any atoms)
        self.__edge_threshold__ = gamma * threshold if atom_type in ('CA', 'CB') else 0.

        # custom
        self.__value__ = np.nan
        self.__gamma__ = gamma
        self.__threshold__ = sps.norm.logpdf(PARAMS['threshold'])
        self.__prior__ = self.__threshold__

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
    def assigned(self):

        return ~np.isnan(self.__value__)

    def assign(self, v):

        self.__value__ = v
        if ~np.isnan(self.mu) and ~np.isnan(self.value):
            self.__prior__ = sps.norm.logpdf((v-self.mu)/self.std)

    def clear(self):

        self.__value__ = np.nan

    def merge(self, other):

        atom = self.copy()

        if self.assigned and other.assigned:
            atom.assign((self.value + other.value) / 2.)
        elif other.assigned:
            atom.assign(other.value)

        return atom

    def node_score(self):

        return self.__prior__

        # return self.__prior__ if self.__prior__ > self.__node_threshold__ else -np.inf

    def edge_score(self, other):

        if self.assigned and other.assigned:
            return self.__gamma__ * sps.norm.logpdf((self.value-other.value)/self.qstd)

        return self.__edge_threshold__


######################################
# ATOM GROUP #########################
######################################

class AtomGroupSS(AtomGroup):

    """
    A group of atoms (typically atoms corresponding to one observable spin
    system or to a contiguous fragment).

    These can be added together to form longer fragments.
    """

    def __init__(self, sequence, atoms, group_residue, gamma=1.,
                 *varargs, **kwargs):

        """
        residue = first residue in the fragment
        """

        super(AtomGroupSS, self).__init__(*varargs, **kwargs)

        self.gamma = gamma

        for atom_type, residue, atom_id in atoms:

            self[atom_id] = AtomSS(atom_type, residue, atom_id, sequence, gamma)
            self.__idkeys__[atom_type, residue] = atom_id

        self.residue = group_residue
        self._update_residues()

        # compute thresholds
        self.__node_threshold__ = np.sum(
            [a.node_threshold for a in self.values()])
        self.__edge_threshold__ = np.sum(
            [a.edge_threshold for a in self.values()]) + self.__node_threshold__

    def _update_residues(self):

        residues = set([a.residue for a in self.values()
                        if a.atom_type in ('N', 'H')])
        self.residues = sorted(list(residues))
        if len(self.residues) == 0 and self.residue is not None:
            self.residues = [self.residue]

    def node_score(self):

        return np.sum([a.node_score() for a in self.values()])

    def edge_score(self, other):

        score = self.node_score()

        common = self.intersection(other)
        for node_no in common:
            score += self[node_no].edge_score(other[node_no])

        return score

    def apply_spin_system(self, spin_system, residue):

        self.__u__ = spin_system.u
        if len(self.residues) == 0:
            self.residues.append(residue)

        for (atom_type, shift), value in zip(SPIN_SYSTEM_ORDER, spin_system.value):
            cur_residue = residue + shift
            try:
                self[atom_type, cur_residue].assign(value)
            except KeyError:
                # print 'atom not found: {} {}'.format(cur_residue, atom_type)
                continue

        self.empty = spin_system.empty
        self.spin_id = spin_system.id if not self.empty else None


######################################
# SPIN SYSTEMS #######################
######################################

class SpinSystem(object):

    def __init__(self, value, i, u, empty=False):

        self.value = value
        if empty:
            self.value *= np.nan
            self.empty = True
        else:
            self.empty = False
        self.__u__ = u
        self.__id__ = i

    @property
    def u(self):

        return self.__u__

    @property
    def id(self):

        return self.__id__

    def __repr__(self):

        return 'SpinSystem({}, id={})'.format(
            super(SpinSystem, self).__repr__(), self.id)

    def __str__(self):

        return 'SpinSystem({}, id={})'.format(
            super(SpinSystem, self).__str__(), self.id)


class SpinSystemSet(list):

    def __init__(self, experiment):

        super(SpinSystemSet, self).__init__()

        data = experiment.spin_systems

        u_max = data.shape[0]
        for i, values in enumerate(data):
            u = np.zeros(u_max, dtype=np.int32)
            u[i] = 1
            self.append(SpinSystem(values, i, u))

        # add empty spin system
        u = np.zeros(u_max, dtype=np.int32)
        self.append(SpinSystem(np.ones(values.shape)*np.nan, i+1, u, empty=True))

        self.__value__ = data

    @property
    def value(self):

        return self.__value__


######################################
# ASSIGNER ###########################
######################################

class AssignerSS(Assigner):

    def __init__(self, experiment, gamma=1., *varargs, **kwargs):

        self.gamma = gamma

        super(AssignerSS, self).__init__(experiment, *varargs, **kwargs)

    def _setup(self):

        self.spin_systems = SpinSystemSet(self.experiment)
        self.peaks = ExpectedPeaks(self.experiment)
        self.atoms = [(t[0], t[1], i) for i, t in enumerate(self.experiment.atoms)]
        self.residues = [self.peaks.subset_to(i).atom_list for i in range(self.n)]

    def _create_nodes(self, verbose=False):

        print '::: Creating nodes'
        for i, atom_list in enumerate(self.residues):
            if verbose:
                print '    ...residue {}'.format(i)
            atoms = [self.atoms[j] for j in atom_list]

            for s in self.spin_systems:
                node = AtomGroupSS(self.sequence, atoms, group_residue=i, gamma=self.gamma)
                node.apply_spin_system(s, i)
                # TODO change to threshold
                if node.node_score() < node.node_threshold:
                    continue
                self.add(node)

            # # free node
            # node = AtomGroupSS(self.sequence, atoms, group_residue=i)
            # self.add(node)

        # start and end nodes
        self.start = AtomGroupSS(self.sequence, [('C', 0, -1)], group_residue=None)
        self.start.apply_spin_system(self.spin_systems[-1], 0)
        self.add(self.start)
        self.end = AtomGroupSS(self.sequence, [('C', self.n-1, -2)], group_residue=None)
        self.end.apply_spin_system(self.spin_systems[-1], self.n-1)
        self.add(self.end)

    def _create_edges(self, verbose=False):

        min_weight = np.inf

        print '::: Creating edges'
        for i in range(self.n-1):
            if verbose:
                print '   ...layer {} -> {}'.format(i, i+1)
            layer = self.layers[i]
            for node in layer:
                o_layer = self.layers[node.residues[-1]+1]
                for o_node in o_layer:
                    score = node.edge_score(o_node)
                    # TODO change to threshold
                    if score < node.edge_threshold:
                        continue
                    self.add_edge(node, o_node)
                    self[node][o_node]['score'] = node.edge_score(o_node)
                    self[node][o_node]['weight'] = -self[node][o_node]['score']
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

    def recall_precision(self, path, true_x, n_manual):

        """
        Precision and Recall.
        """

        n_correct = 0
        n_assigned = 0

        for i, node in enumerate(path[1:-1]):
            if node.empty:
                continue
            if len(node) == 0:
                continue
            n_assigned += 1
            correct = 0
            count = 0
            for atom_no, atom in node.iteritems():
                if np.isnan(atom.value):
                    continue
                if np.isnan(true_x[atom_no]):
                    continue
                count += 1
                error = np.abs(atom.value - true_x[atom_no])
                correct += int(error < atom.tolerance)

            # condition for correctness
            if float(count) / len(node) > 0.5 and float(correct) / len(node) > 0.75:
                n_correct += 1

        recall = float(n_correct) / n_manual
        precision = float(n_correct) / n_assigned
        print 'Assigned {} spin systems.'.format(n_assigned)
        print 'Correctly assigned {} spin systems'.format(n_correct)
        print 'Recall = {}\nPrecision={}'.format(
            float(n_correct) / n_manual,
            float(n_correct) / n_assigned)

        return recall, precision

    def best_spin_system(self, true_x, i, t=9.):

        """
        Best spin system for residue i.
        """

        atoms = self.layers[i][0]

        distances = np.zeros(len(self.spin_systems)) * np.nan

        for k, spin in enumerate(self.spin_systems):
            values = np.ones(6) * np.nan
            true = np.ones(6) * np.nan
            qvar = np.ones(6) * np.nan
            for j, (atom_type, d) in enumerate(SPIN_SYSTEM_ORDER):
                values[j] = spin.value[j]
                try:
                    true[j] = true_x[atoms[atom_type, i+d].id]
                    qvar[j] = atoms[atom_type, i+d].qvar
                except KeyError:
                    continue
            diffs = (true-values)**2/qvar
            if np.all(np.isnan(diffs)):
                distances[k] = np.nan
                continue
            distances[k] = np.nanmean((true-values)**2/qvar)

        distance = np.nanmin(distances)
        threshold = np.sum(~np.isnan(distances)) * t
        if distance < threshold:
            return np.nanargmin(distances)
        return np.nan

    def n_correct(self, path, true_x, threshold=9.):

        """
        Number of correct spin systems in path.
        """

        correct = [self.best_spin_system(true_x, i, threshold)
                   for i in range(self.n)]
        assigned = [node.spin_id for node in path[1:-1]]
        n_empty = np.sum([node.empty for node in path[1:-1]])

        n_assignable = np.sum(~np.isnan(correct))
        n_assigned = len(assigned) - n_empty
        n_correct = np.sum(np.array(correct) == np.array(assigned))

        print '# assignable = {}'.format(n_assignable)
        print '# assigned = {}'.format(n_assigned)
        print '# correct = {}'.format(n_correct)
        print 'PRC/RCL = {}/{}'.format(n_correct/float(n_assigned),
                                       n_correct/float(n_assignable))

        return correct, assigned

    def ordered_spins(self, e):

        """
        Returns ordered spin systems.
        """

        spins = [e.spin_systems[node.spin_id, :] if node.spin_id is not None
                 else e.spin_systems[0, :] * np.nan for node in self.path[1:-1]]
        spins = [s.reshape((1, -1)) for s in spins]
        return np.concatenate(spins, axis=0)

