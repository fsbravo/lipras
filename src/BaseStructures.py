"""
Base classes for Assigner.
"""

from common import *

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

import gurobipy as gbp


######################################
# BASE ATOM CLASS ####################
######################################

class Atom(object):

    """
    Basic interface for an atom.
    """

    def __init__(self):

        self.__residue__ = None
        self.__atom_type__ = None
        self.__residue_type__ = None

        self.__mu__ = np.nan
        self.__var__ = np.nan
        self.__std__ = np.nan
        self.__qvar__ = np.nan
        self.__qstd__ = np.nan
        self.__tolerance__ = np.nan

        self.__node_threshold__ = -np.inf
        self.__edge_threshold__ = -np.inf

        self.__value__ = np.nan
        self.__id__ = None

        self.__box__ = None

    def __str__(self):

        return 'Atom({}{} ({}))'.format(self.residue,
                                        self.atom_type,
                                        self.residue_type)

    def __repr__(self):

        return 'Atom({}{} ({}))'.format(self.residue,
                                        self.atom_type,
                                        self.residue_type)

    @property
    def id(self):

        return self.__id__

    @property
    def residue(self):

        return self.__residue__

    @property
    def atom_type(self):

        return self.__atom_type__

    @property
    def residue_type(self):

        return self.__residue_type__

    @property
    def mu(self):

        return self.__mu__

    @property
    def std(self):

        return self.__std__

    @property
    def var(self):

        return self.__var__

    @property
    def qvar(self):

        return self.__qvar__

    @property
    def qstd(self):

        return self.__qstd__

    @property
    def tolerance(self):

        return self.__tolerance__

    @property
    def node_threshold(self):

        return self.__node_threshold__

    @property
    def edge_threshold(self):

        return self.__edge_threshold__

    def copy(self):

        return copy.deepcopy(self)

    @property
    def bounding_box(self):

        if self.assigned:
            mu = self.value
            tail = self.tolerance
        else:
            mu = self.mu
            tail = self.std * PARAMS['threshold']

        return mu-tail, mu+tail

    @property
    def lower(self):

        if self.assigned:
            return self.value - self.tolerance
        return self.raw_lower

    @property
    def upper(self):

        if self.assigned:
            return self.value + self.tolerance
        return self.raw_upper

    @property
    def raw_lower(self):

        return self.mu - self.std * PARAMS['threshold']

    @property
    def raw_upper(self):

        return self.mu + self.std * PARAMS['threshold']

    @property
    def box(self):

        if self.__box__ is None:
            self.__box__ = Box(self.raw_lower, self.raw_upper)

        return self.__box__

    @property
    def value(self):

        raise NotImplementedError()

    @property
    def assigned(self):

        raise NotImplementedError()

    def assign(self, value):

        raise NotImplementedError()

    def clear(self):

        raise NotImplementedError()

    def merge(self, other):

        raise NotImplementedError()

    def node_score(self):

        raise NotImplementedError()

    def edge_score(self):

        raise NotImplementedError()


######################################
# BASE ATOM GROUP ####################
######################################

class AtomGroup(OrderedDict):

    """
    A group of atoms (typically atoms corresponding to one observable spin
    system, but can also correspond to a contiguous fragment).

    These can be added together to form longer fragments.
    """

    _ids = count(0)

    def __init__(self, *varargs, **kwargs):

        super(AtomGroup, self).__init__(*varargs, **kwargs)

        self.__id__ = next(self._ids)
        self.__idkeys__ = {}
        self.__u__ = None
        self.__node_threshold__ = -np.inf
        self.__edge_threshold__ = -np.inf

    def __hash__(self):

        return self.id

    def __getitem__(self, i):

        if isinstance(i, tuple):
            return super(AtomGroup, self).__getitem__(self.__idkeys__[i[0], i[1]])

        return super(AtomGroup, self).__getitem__(i)

    @property
    def id(self):

        return self.__id__

    @property
    def u(self):

        return self.__u__

    @property
    def value(self):

        return np.array([a.value for a in self.values()])

    @property
    def mu(self):

        return np.array([a.mu for a in self.values()])

    @property
    def var(self):

        return np.array([a.var for a in self.values()])

    @property
    def std(self):

        return np.array([a.std for a in self.values()])

    @property
    def qstd(self):

        return np.array([a.qstd for a in self.values()])

    @property
    def qvar(self):

        return np.array([a.qvar for a in self.values()])

    @property
    def tolerance(self):

        return np.array([a.tolerance for a in self.values()])

    @property
    def node_threshold(self):

        return self.__node_threshold__

    @property
    def edge_threshold(self):

        return self.__edge_threshold__

    def print_atom_values(self):

        for atom in self.values():
            print '{} = {}'.format(str(atom), atom.value)

    def copy(self):

        return copy.deepcopy(self)

    def intersection(self, other):

        keys1 = set(self.keys())
        keys2 = set(other.keys())

        return keys1.intersection(keys2)

    def union(self, other):

        keys1 = set(self.keys())
        keys2 = set(other.keys())

        return keys1.union(keys2)

    def __add__(self, other):

        keys1 = set(self.keys())
        keys2 = set(other.keys())
        common = keys1.intersection(keys2)
        group1 = keys1 - keys2
        group2 = keys2 - keys1

        items = []
        items += [(atom_id, self[atom_id].copy()) for atom_id in group1]
        items += [(atom_id, self[atom_id].merge(other[atom_id])) for atom_id in common]
        items += [(atom_id, other[atom_id].copy()) for atom_id in group2]

        return AtomGroup(None, sorted(items, key=lambda t: t[0]))

    def node_score(self):

        raise NotImplementedError()

    def edge_score(self, other):

        raise NotImplementedError()


######################################
# BASE ASSIGNER ######################
######################################

class Assigner(nx.DiGraph):

    """
    Base Assigner class for LP based solutions.

    Subclasses must implement:

        _setup
        _create_nodes
        _create_edges
    """

    def __init__(self, experiment, *varargs, **kwargs):

        super(Assigner, self).__init__(*varargs, **kwargs)

        self.experiment = experiment
        self.sequence = experiment.sequence
        self.__n__ = experiment.size

        self.layers = [[] for _ in range(self.__n__)]

        self._setup()
        self._create_nodes()
        self._create_edges()
        self.__u__ = None

    def add(self, node):

        self.add_node(node)
        if node.residue is not None:
            self.layers[node.residue].append(node)
        else:
            print 'no layer! {}'.format(node)

    @property
    def u(self):

        return self.__u__

    @property
    def n(self):

        return self.__n__

    def _setup(self):

        """
        Initialize data structures required for creation of nodes.
        """

        raise NotImplementedError()

    def _create_nodes(self):

        """
        Create all nodes.
        """

        raise NotImplementedError()

    def _create_edges(self):

        """
        Create all edges.
        """

        raise NotImplementedError()

    def draw(self, g, spacing=5., *varargs, **kwargs):

        """
        Uses matplotlib to draw graph.
        """

        # layers
        layers = [[] for _ in range(self.n)]
        start = None
        end = None
        for node in g.nodes():
            if node.residue is not None:
                layers[node.residue].append(node)
            if node.id == self.start.id:
                start = node
            elif node.id == self.end.id:
                end = node

        # count nodes per residue
        counts = [len(l) for l in layers]

        # node vertical positions for each layer
        pos = [(np.arange(c)-np.mean(np.arange(c)))*spacing for c in counts]

        positions = {}
        for i, layer in enumerate(layers):
            p_vec = pos[i]
            for p, node in zip(p_vec, layer):
                positions[node] = [i*spacing, p]
        positions[start] = [-spacing, 0.]
        positions[end] = [self.n*spacing, 0.]

        fig, ax = plt.subplots(figsize=(self.n, np.max(counts)))
        nx.draw(g, pos=positions, ax=ax, *varargs, **kwargs)
        plt.plot()

    def accuracy(self, path, true_x):

        """
        Prints out accuracy for each residue for a given path.
        """

        for i, node in enumerate(path[1:-1]):
            output = Back.WHITE + 'Residue {} '.format(i)
            total_correct = 0
            for atom_no, atom in node.iteritems():
                # if atom.residue != i:
                #     continue
                error = np.round(atom.value-true_x[atom_no], 3)
                correct = np.abs(error) < atom.tolerance
                total_correct += correct
                if np.isnan(error):
                    output += Back.WHITE + str(error) + ' '
                else:
                    if correct:
                        output += Back.GREEN + str(error) + ' '
                    else:
                        output += Back.RED + str(error) + ' '
            output += '\t{}'.format(total_correct)

            print output

    def optimize(self, G=None, max_utilization=1, max_variable=1):

        """
        Builds an integer linear program to find a final path within solution graph
        """

        if G is None:
            G = self

        model = gbp.Model('LP')

        # ### fix edge order
        print '...fixing edge order...'
        edge_order = sorted(G.edges())
        e_translator = {n: i for i, n in enumerate(edge_order)}
        e_translator_r = {i: n for i, n in enumerate(edge_order)}

        # ### variable
        print '...creating variable...'
        n_edges = len(G.edges())
        x = [model.addVar(obj=self[t[0]][t[1]]['score'], lb=0, ub=max_variable)
             for t in edge_order]

        print '...created binary variable of size {}!'.format(n_edges)

        # ### constraints
        print '...assembling constraints'
        # path constraints
        print '   ...path constraints...'
        for i, node in enumerate(G.nodes()):
            pred = G.pred[node]
            pred_edges = [e_translator[(p, node)] for p in pred]
            succ = G.succ[node]
            succ_edges = [e_translator[(node, s)] for s in succ]
            tmp = None
            if len(pred_edges) == 0 and len(succ_edges) == 0:
                continue
            if len(pred_edges) == 0:
                tmp = x[succ_edges[0]]
                for j in succ_edges[1:]:
                    tmp += x[j]
            else:
                tmp = -x[pred_edges[0]]
                for j in pred_edges[1:]:
                    tmp -= x[j]
                for j in succ_edges:
                    tmp += x[j]
            if tmp is None:
                print 'tmp is None and something is wrong!!!'
            else:
                print 'tmp = ', tmp
            # variable to constrain
            if node.id == self.start.id:
                model.addConstr(tmp == 1, 'start')
            elif node.id == self.end.id:
                model.addConstr(tmp == -1, 'end')
            else:
                model.addConstr(tmp == 0, 'eq' + str(i))

        # utilization constraints
        print '   ...utilization constraints...'
        u_vecs = [edge[1].u for edge in edge_order]
        for i in self.u:
            idx = [i for i, u in enumerate(u_vecs) if i in u]
            if len(idx) > 0:
                tmp = x[idx[0]]
                for j in idx[1:]:
                    tmp += x[j]
            model.addConstr(tmp <= 1, 'u' + str(i))

        # ### objective
        print '...assembling objective'
        scores = np.array([self[t[0]][t[1]]['score'] for t in edge_order])
        obj = scores[0] * x[0]
        for i, s in enumerate(scores[1:]):
            obj += s * x[i+1]
        model.setObjective(obj, gbp.GRB.MAXIMIZE)

        # ### solve
        print '... solving ...'
        model.optimize()
        print '... done!'

        # ### build solution path
        print '... building solution path ...'
        x_val = np.array([var.x for var in x])
        e_index = np.nonzero(x_val > 0.5)[0]
        edges = [edge_order[i] for i in e_index]
        self.build_solution_graph(edges)
        print '... done!'

    def build_solution_graph(self, edges):

        """
        Build new graph from solution
        """

        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        for layer in self.layers:
            nodes.add(layer[-1])
        self.solution_graph = self.subgraph(nodes)

    def optimize_lagrangian(self, G=None, max_utilization=1, max_variable=1,
                            lambda_p=30., lambda_m=1.):

        """
        Solve LP to generate solution path.
        """

        if G is None:
            G = self

        # ### fix edge order
        print '...fixing edge order'
        self.edge_order = sorted(G.edges())
        self.e_translator = {n: i for i, n in enumerate(self.edge_order)}
        self.e_translator_r = {i: n for i, n in enumerate(self.edge_order)}

        # ### variable
        print '...creating variable'
        n_edges = len(G.edges())
        self.variable = Variable((1, n_edges))

        print self.variable.shape

        # ### constraints
        print '...assembling constraints'
        constraints = []
        # path constraints
        for node in G.nodes():
            pred = G.pred[node]
            pred_edges = [self.e_translator[(p, node)] for p in pred]
            succ = G.succ[node]
            succ_edges = [self.e_translator[(node, s)] for s in succ]
            # variable to constrain
            succ_sum = sum(self.variable[0, succ_edges]) if len(succ_edges) > 0 else 0
            pred_sum = sum(self.variable[0, pred_edges]) if len(pred_edges) > 0 else 0
            if node.id == self.start.id:
                c = succ_sum - pred_sum == 1
            elif node.id == self.end.id:
                c = succ_sum - pred_sum == -1
            else:
                c = succ_sum - pred_sum == 0
            constraints.append(c)
        # utilization constraints
        u_vecs = [np.zeros((1, len(self.u))) for edge in self.edge_order]
        for i, edge in enumerate(self.edge_order):
            u_vecs[i][edge[1].u] = 1
        print u_vecs[0].shape
        u = np.concatenate(u_vecs, axis=0)
        print u.shape
        self.utilization = self.variable * u
        self.epsilon_p = Variable(self.utilization.shape)
        self.epsilon_m = Variable(self.utilization.shape)
        constraints.append(self.epsilon_p-self.epsilon_m+self.utilization == max_utilization)
        constraints.append(self.epsilon_p >= 0)
        constraints.append(self.epsilon_m >= 0)
        # positivity and stochasticity constraints
        constraints.append(self.variable >= 0)
        constraints.append(self.variable <= max_variable)

        # ### cost
        print '...assembling cost'
        scores = np.array([self[t[0]][t[1]]['weight'] for t in self.edge_order])
        scores.reshape((-1, 1))
        cost = sum(self.variable * scores) + lambda_p * sum(self.epsilon_p) + lambda_m * sum(self.epsilon_m)

        # ### problem
        self.problem = Problem(Minimize(cost), constraints)

        # ### solve
        print '... solving ...'
        self.problem.solve(solver=GUROBI)
        print '... done!'

        # ### build solution graph
        print '... building solution graph ...'
        e_index = np.nonzero(self.variable.value > 0.05)[1]
        edges = [self.edge_order[i] for i in e_index]
        self.build_solution_path(edges)
        print '... done!'

    def build_solution_path(self, edges):

        """
        Builds path from solution
        """

        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        self.solution_path = self.subgraph(nodes)
        nodes = sorted(list(self.solution_path.nodes()), key=lambda node: node.residue)
        self.path = [nodes[0]] + nodes[2:] + [nodes[1]]
        # self.path = nx.shortest_path(self.solution_path, source=self.start, target=self.end, weight='weight')

    def optimize_ilp(self, G=None):

        """
        Builds an integer linear program to find a final path within solution graph
        """

        if G is None:
            G = self.solution_graph

        model = gbp.Model('ILP')

        # ### fix edge order
        print '...fixing edge order...'
        edge_order = sorted(G.edges())
        e_translator = {n: i for i, n in enumerate(edge_order)}
        e_translator_r = {i: n for i, n in enumerate(edge_order)}

        # ### variable
        print '...creating variable...'
        n_edges = len(G.edges())
        x = [model.addVar(obj=self[t[0]][t[1]]['score'], vtype=gbp.GRB.BINARY)
             for t in edge_order]

        print '...created binary variable of size {}!'.format(n_edges)


        # ### constraints
        print '...assembling constraints'
        # path constraints
        print '   ...path constraints...'
        for i, node in enumerate(G.nodes()):
            pred = G.pred[node]
            pred_edges = [e_translator[(p, node)] for p in pred]
            succ = G.succ[node]
            succ_edges = [e_translator[(node, s)] for s in succ]
            tmp = None
            if len(pred_edges) == 0 and len(succ_edges) == 0:
                continue
            if len(pred_edges) == 0:
                tmp = x[succ_edges[0]]
                for j in succ_edges[1:]:
                    tmp += x[j]
            else:
                tmp = -x[pred_edges[0]]
                for j in pred_edges[1:]:
                    tmp -= x[j]
                for j in succ_edges:
                    tmp += x[j]
            # variable to constrain
            if node.id == self.start.id:
                model.addConstr(tmp == 1, 'start')
            elif node.id == self.end.id:
                model.addConstr(tmp == -1, 'end')
            else:
                model.addConstr(tmp == 0, 'eq' + str(i))

        # utilization constraints
        print '   ...utilization constraints...'
        u_vecs = [edge[1].u for edge in edge_order]
        for i in self.u:
            idx = [i for i, u in enumerate(u_vecs) if i in u]
            if len(idx) > 0:
                tmp = x[idx[0]]
                for j in idx[1:]:
                    tmp += x[j]
            model.addConstr(tmp <= 1, 'u' + str(i))

        # ### objective
        print '...assembling objective'
        scores = np.array([self[t[0]][t[1]]['score'] for t in edge_order])
        obj = scores[0] * x[0]
        for i, s in enumerate(scores[1:]):
            obj += s * x[i+1]
        model.setObjective(obj, gbp.GRB.MAXIMIZE)

        # ### solve
        print '... solving ...'
        model.optimize()
        print '... done!'

        # ### build solution path
        print '... building solution path ...'
        x_val = np.array([var.x for var in x])
        e_index = np.nonzero(x_val > 0.5)[0]
        edges = [edge_order[i] for i in e_index]
        self.build_solution_path(edges)
        print '... done!'
