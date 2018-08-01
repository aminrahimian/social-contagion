from settings import *

import settings

# if settings.do_plots:
#     import matplotlib.pyplot as plt
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     from decimal import Decimal
#     FOURPLACES = Decimal(10) ** -4
#     ERROR_BAR_TYPE = 'std'

import torch
import pandas as pd
import copy

import networkx as NX

from scipy.stats import norm

susceptible = 0
infected = 1

SIMPLE = 0
COMPLEX = 1

SENTINEL = object()


def random_factor_pair(value):
    """
    Returns a random pair that factor value.
    It is used to set the number of columns and rows in  2D grid with a given size such that
    size = num_columns*num_rows
    """
    factors = []
    for i in range(1, int(value**0.5)+1):
        if value % i == 0:
            factors.append((int(i), value // i))
    return RD.choice(factors)


def time_the_cap(fraction_timeseries, cap):
    """
    Returns the first time a time-series of the fractions goes above the cap.
    Extrapolates if too slow, or returns -1 if not-spread
    """
    time_of_the_cap = None
    total_steps = len(fraction_timeseries)
    if fraction_timeseries[-1] < cap:
        if fraction_timeseries[0] == fraction_timeseries[-1]:  # contagion has not spread
            time_of_the_cap = -1
        else:
            count_steps = 0
            for i in reversed(range(total_steps)):
                if fraction_timeseries[i] < fraction_timeseries[-1]:
                    count_steps += 1
                    time_of_the_cap = ((cap - fraction_timeseries[-1]) /
                                       (fraction_timeseries[-1] - fraction_timeseries[i]))*count_steps + total_steps - 1
                    break
                    count_steps +=1
    else:
        for i in reversed(range(total_steps)):
            if fraction_timeseries[i] < cap:
                time_of_the_cap = i + 1
                break
    assert time_of_the_cap is not None, 'time_of_the_cap is not set'
    return time_of_the_cap


def newman_watts_add_fixed_number_graph(n, k=2, p=2, seed=None):
    """Returns a Newman–Watts–Strogatz small-world graph. With a fixed (not random) number of edges
    added to each node. Modified newman_watts_strogatzr_graph() in NetworkX.
    """
    if seed is not None:
        RD.seed(seed)
    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    G = NX.connected_watts_strogatz_graph(n, k, 0)
    # G=NX.empty_graph(n)
    # G.name="newman_watts_strogatz_graph(%s,%s,%s)"%(n,k,p)
    # n_list = G.nodes()
    # from_v = list(n_list)
    # # connect the k/2 neighbors
    # for j in range(1, k // 2+1):
    #     to_v = from_v[j:] + from_v[0:j] # the first j are now last
    #     for i in range(len(from_v)):
    #         G.add_edge(from_v[i], to_v[i])
    # # for each node u, randomly select p existing
    # # nodes w and add new edges u-w

    # NX.draw_circular(G)
    # plt.show()

    all_nodes = G.nodes()
    for u in all_nodes:
        count = 0
        while count < p:

            w = np.random.choice(all_nodes)
            # print('w',w)
            # no self-loops and reject if edge u-w exists
            # is that the correct NWS model?
            while w == u or G.has_edge(u, w):
                w = np.random.choice(all_nodes)
                # print('re-drawn w', w)
                if G.degree(u) >= n-1:
                    break # skip this rewiring
            G.add_edge(u, w)
            count += 1
    return G


def cycle_union_Erdos_Renyi(n, k=4, c=2, seed=None):
    """Returns a cycle C_k union G(n,c/n) graph by composing
    NX.connected_watts_strogatz_graph(n, k, 0) and
    NX.erdos_renyi_graph(n, c/n, seed=None, directed=False)"""
    if seed is not None:
        RD.seed(seed)

    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    C_k = NX.connected_watts_strogatz_graph(n, k, 0)

    G_npn = NX.erdos_renyi_graph(n, c/n, seed=None, directed=False)

    assert G_npn.nodes() == C_k.nodes(), "node sets are not the same"
    composed = NX.compose(C_k,G_npn)

    return composed

def c_1_c_2_interpolation(n, eta, add_long_ties_exp, remove_cycle_edges_exp,seed=None):
    """Return graph that interpolates C_1 and C_2.
    Those edges having add_long_ties_exp < eta are added.
    Those edges having remove_cycle_edges_exp < eta are removed.
    len(add_long_ties_exp) = n*(n-1)//2
    len(remove_cycle_edges_exp) = n
    """
    if seed is not None:
        RD.seed(seed)

    assert len(add_long_ties_exp) == n*(n-1)//2, "add_long_ties_exp has the wrong size"
    assert len(remove_cycle_edges_exp) == n, "remove_cycle_edges_exp has the wrong size"

    C_2 = NX.connected_watts_strogatz_graph(n, 4, 0)

    C_2_minus_C_1_edge_index = 0
    removal_list = []

    # NX.draw_circular(C_2)
    # plt.show()

    # print(eta)

    for edge in C_2.edges():
        # print(edge)
        if abs(edge[0] - edge[1]) == 2 or abs(edge[0] - edge[1]) == n-2: # it is a C_2\C_1 edge
            if remove_cycle_edges_exp[C_2_minus_C_1_edge_index] < eta:
                removal_list += [edge]
            C_2_minus_C_1_edge_index += 1 # index the next C_2\C_1 edge
    C_2.remove_edges_from(removal_list)

    # print(removal_list)

    # NX.draw_circular(C_2)
    # plt.show()
    # print(C_2.edges())

    addition_list = []

    K_n = NX.complete_graph(n)

    random_add_edge_index = 0

    for edge in K_n.edges():
        # print(edge)
        if add_long_ties_exp[random_add_edge_index] < eta:
            addition_list += [edge]
        random_add_edge_index += 1 # index the next edge to be considered for addition
    C_2.add_edges_from(addition_list)
    # NX.draw_circular(C_2)
    # plt.show()
    # print(C_2.edges())
    # print(addition_list)
    return C_2


def add_edges(G, number_of_edges_to_be_added=10, mode='random', seed=None):
    """Add number_of_edges_to_be_added edges to the NetworkX object G.
    Two modes: 'random' or 'triadic_closures'
    """
    if seed is not None:
        RD.seed(seed)

    number_of_edges_to_be_added = int(np.floor(number_of_edges_to_be_added))

    assert type(G) is NX.classes.graph.Graph, "input should be a NetworkX graph object"

    fat_network = copy.deepcopy(G)

    # print('original number of edges',len(fat_network.edges()))

    unformed_edges = list(NX.non_edges(fat_network))

    if len(unformed_edges) < number_of_edges_to_be_added:
        print("There are not that many edges left ot be added")
        fat_network.add_edges_from(unformed_edges)  # add all the edges that are left
        return fat_network

    if mode is 'random':
        addition_list = RD.sample(unformed_edges, number_of_edges_to_be_added)
        fat_network.add_edges_from(addition_list)
        return fat_network

    # NX.draw_circular(C_2)
    # plt.show()

    # print(eta)

    if mode is 'triadic_closures':
        weights = []
        for non_edge in unformed_edges:
            weights += [len(list(NX.common_neighbors(G, non_edge[0], non_edge[1])))]
        # print('weights',weights)
        total_sum = sum(weights)
        # print('total_sum',total_sum)
        normalized_weights = [weight/total_sum for weight in weights]

        # print('normalized_weights', normalized_weights)

        # print('unformed_edges', unformed_edges)

        addition_list = np.random.choice(range(len(unformed_edges)),
                                         number_of_edges_to_be_added,
                                         replace=False,
                                         p=normalized_weights)

        addition_list = addition_list.astype(int)

        # print(addition_list)

        addition_list = [unformed_edges[ii] for ii in list(addition_list)]
        # print('addition_list', addition_list)
        fat_network.add_edges_from(addition_list)
        # print(fat_network.edges())
        return fat_network

    # for edge in C_2.edges():
    #     # print(edge)
    #     if abs(edge[0] - edge[1]) == 2 or abs(edge[0] - edge[1]) == n-2: # it is a C_2\C_1 edge
    #         if remove_cycle_edges_exp[C_2_minus_C_1_edge_index] < eta:
    #             removal_list += [edge]
    #         C_2_minus_C_1_edge_index += 1 # index the next C_2\C_1 edge
    # C_2.remove_edges_from(removal_list)
    #
    # # print(removal_list)
    #
    # # NX.draw_circular(C_2)
    # # plt.show()
    # # print(C_2.edges())
    #
    # addition_list = []
    #
    # K_n = NX.complete_graph(n)
    #
    # random_add_edge_index = 0
    #
    # for edge in K_n.edges():
    #     # print(edge)
    #     if add_long_ties_exp[random_add_edge_index] < eta:
    #         addition_list += [edge]
    #     random_add_edge_index += 1 # index the next edge to be considered for addition
    # C_2.add_edges_from(addition_list)
    # # NX.draw_circular(C_2)
    # # plt.show()
    # # print(C_2.edges())
    # # print(addition_list)
    # return C_2

class network_model():
    """
    implement the initializations and parameter set methods
    """
    def __init__(self):
        pass

    def init_network(self):
        """
        initializes the network interconnections based on the params
        """
        if 'network' in self.fixed_params:
            self.params['network'] = self.fixed_params['network']
        elif 'network' not in self.fixed_params:
            if self.params['network_model'] == 'erdos_renyi':
                if 'linkProbability' not in self.fixed_params:  # erdos-renyi link probability
                    self.params['linkProbability'] = 2 * np.log(self.params['size']) / self.params[
                        'size']  # np.random.beta(1, 1, None)*20*np.log(self.params['size'])/self.params['size']
                self.params['network'] = NX.erdos_renyi_graph(self.params['size'], self.params['linkProbability'])
                if not NX.is_connected(self.params['network']):
                    self.params['network'] = NX.erdos_renyi_graph(self.params['size'], self.params['linkProbability'])

            elif self.params['network_model'] == 'watts_strogatz':
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 3
                if 'rewiring_probability' not in self.fixed_params:
                    self.params['rewiring_probability'] = 0.000000005
                self.params['network'] = NX.connected_watts_strogatz_graph(self.params['size'], self.params['nearest_neighbors'],
                                                                 self.params['rewiring_probability'])
            elif self.params['network_model'] == 'grid':
                if 'number_grid_rows' not in self.fixed_params:
                    if 'number_grid_columns' not in self.fixed_params:
                        (self.params['number_grid_columns'],self.params['number_grid_rows']) = random_factor_pair(self.params['size'])
                    else:
                        self.params['number_grid_rows'] = self.params['size'] // self.params['number_grid_columns']
                        self.params['number_grid_columns'] = self.params['size'] // self.params['number_grid_rows']
                elif 'number_grid_columns' in self.fixed_params:
                    assert self.params['number_grid_columns']*self.params['number_grid_rows'] == self.params['size'], \
                        'incompatible size and grid dimensions'
                else:
                    self.params['number_grid_columns'] = self.params['size'] // self.params['number_grid_rows']
                    self.params['number_grid_rows'] = self.params['size'] // self.params['number_grid_columns']
                self.params['network'] = NX.grid_2d_graph(self.params['number_grid_rows'],
                                                          self.params['number_grid_columns'])
            elif self.params['network_model'] == 'random_regular':
                if 'degree' not in self.fixed_params:
                    self.params['degree'] = np.random.randint(1, 6)
                self.params['network'] = NX.random_regular_graph(self.params['degree'], self.params['size'], seed=None)
            elif self.params['network_model'] == 'newman_watts_fixed_number':
                if 'fixed_number_edges_added' not in self.fixed_params:
                    self.params['fixed_number_edges_added'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = newman_watts_add_fixed_number_graph(self.params['size'], self.params['nearest_neighbors']
                                                                             , self.params['fixed_number_edges_added'])
            elif self.params['network_model'] == 'cycle_union_Erdos_Renyi':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = cycle_union_Erdos_Renyi(self.params['size'], self.params['nearest_neighbors'],
                                                                 self.params['c'])
                # NX.draw_circular(self.params['network'])
                # plt.show()

            elif self.params['network_model'] == 'c_1_c_2_interpolation':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = c_1_c_2_interpolation(self.params['size'],self.params['eta'],
                                                               self.params['add_long_ties_exp'],
                                                               self.params['remove_cycle_edges_exp'])
                # NX.draw_circular(self.params['network'])
                # plt.show()
            else:
                assert False, 'undefined network type'

        # print('we are here')
        # print(self.fixed_params)

        # additional modifications to the network topology which include rewiring
        # and edge additions


        if 'rewire' not in self.fixed_params:
            self.params['rewire'] = False
            print('warning: the network will not be rewired!')

        if self.params['rewire']:

            if 'rewiring_mode' not in self.fixed_params:
                self.params['rewiring_mode'] = 'maslov_sneppen'
                print('warning: the rewiring mode is set to maslov_sneppen')
            if self.params['rewiring_mode'] == 'maslov_sneppen':
                if 'num_steps_for_maslov_sneppen_rewiring' not in self.fixed_params:
                    self.params['num_steps_for_maslov_sneppen_rewiring'] = \
                        0.1 * self.params['network'].number_of_edges()  # rewire 10% of edges
                    print('Warning: num_steps_for_maslov_sneppen_rewiring is set to default 10%')
                rewired_network = \
                    self.maslov_sneppen_rewiring(
                        num_steps=int(np.floor(self.params['num_steps_for_maslov_sneppen_rewiring'])))
            elif self.params['rewiring_mode'] == 'random_random':
                if 'num_edges_for_random_random_rewiring' not in self.fixed_params:
                    self.params['num_edges_for_random_random_rewiring'] = \
                        0.1 * self.params['network'].number_of_edges()  # rewire 10% of edges
                    print('warning: num_edges_for_random_random_rewiring is set to default 10%')

                rewired_network = \
                    self.random_random_rewiring(
                        num_edges=int(np.floor(self.params['num_edges_for_random_random_rewiring'])))

        # if 'maslov_sneppen' not in self.fixed_params:
        #     self.params['maslov_sneppen'] = False
        #     # print('we did not rewire!')
        # if self.params['maslov_sneppen']:
        #     if 'num_steps_for_maslov_sneppen_rewiring' not in self.fixed_params:
        #         self.params['num_steps_for_maslov_sneppen_rewiring'] = \
        #             0.1 * self.params['network'].number_of_edges()  # rewire 10% of edges
        #     rewired_network = self.maslov_sneppen_rewiring(num_steps=
        #                                                    int(np.floor(
        #                                                        self.params['num_steps_for_maslov'
        #                                                                    '_sneppen_rewiring'])))

            # print('we rewired!')
            # while (not NX.is_connected(rewired_network)):
            #     rewired_network = self.maslov_sneppen_rewiring(num_steps=
            #                                  int(np.floor(self.params['num_steps_for_maslov_sneppen_rewiring'])))
            #     print('we rewired!')

            self.params['network'] = rewired_network
            # print(self.params['network'].number_of_edges())

            # print(NX.is_connected(self.params['network']))

        if 'add_edges' not in self.fixed_params:
            self.params['add_edges'] = False

        if self.params['add_edges']:
            if 'edge_addition_mode' not in self.fixed_params:
                self.params['edge_addition_mode'] = 'triadic_closures'
            if 'number_of_edges_to_be_added' not in self.fixed_params:
                self.params['number_of_edges_to_be_added'] = \
                    int(np.floor(0.15 * self.params['network'].number_of_edges()))  # add 15% more edges

            fattened_network = add_edges(self.params['network'],
                                         self.params['number_of_edges_to_be_added'],
                                         self.params['edge_addition_mode'])

            # print('we fat!')
                # while (not NX.is_connected(rewired_network)):
                #     rewired_network = self.maslov_sneppen_rewiring(num_steps=
                #                                  int(np.floor(self.params['num_steps_for_maslov_sneppen_rewiring'])))
                #     print('we rewired!')

            self.params['network'] = fattened_network

                # print(NX.is_connected(self.params['network']))

    def init_network_states(self):
        """
        initializes the node states (infected/susceptible) and other node attributes such as number of infected neighbors
        and time since infection
        """

        self.number_of_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False

        if 'thresholds' not in self.fixed_params:
            if hasattr(self,'isLinearThresholdModel'):
                relative_thresholds = list(np.random.uniform(size=self.params['size']))
                all_degrees = self.params['network'].degree()
                self.params['thresholds'] = list(map(lambda x, y: float(x[1]) * y, all_degrees, relative_thresholds))
            else:
                self.params['thresholds'] = [self.params['theta']] * self.params['size']

        if 'initial_states' not in self.fixed_params:
            if 'initialization_mode' not in self.fixed_params:
                self.params['initialization_mode'] = 'fixed_probability_initial_infection'
                print('warning: The initialization_mode not supplied, '
                      'set to default fixed_probability_initial_infection')
            if self.params['initialization_mode'] is 'fixed_probability_initial_infection':
                # 'initial_infection_probability' should be specified.
                if 'initial_infection_probability' not in self.fixed_params:
                    self.params['initial_infection_probability'] = 0.1
                    print('warning: The initial_infection_probability not supplied, set to default 0.1')

                self.params['initial_states'] = np.random.binomial(1, [self.params['initial_infection_probability']]*
                                                                   self.params['size'])
            elif self.params['initialization_mode'] is 'fixed_number_initial_infection':

                if 'initial_infection_number' not in self.fixed_params:
                    self.params['initial_infection_number'] = 2
                    print('warning: The initial_infection_not supplied, set to default 2.')

                initially_infected_node_indexes = np.random.choice(range(self.params['size']),
                                                                   self.params['initial_infection_number'],
                                                                   replace=False)
                self.params['initial_states'] = 1.0*np.zeros(self.params['size'])
                self.params['initial_states'][initially_infected_node_indexes] = 1.0
                self.params['initial_states'] = list(self.params['initial_states'])
                # print(self.params['initial_states'])
            else:
                assert False, "undefined initialization_mode"

        count_nodes = 0
        for i in self.params['network'].nodes():
            self.params['network'].node[i]['number_of_infected_neighbors'] = 0
            self.params['network'].node[i]['time_since_infection'] = 0
            self.params['network'].node[i]['threshold'] = self.params['thresholds'][count_nodes]
            count_nodes += 1

        self.time_since_infection_is_updated = True

        count_nodes = 0
        for i in self.params['network'].nodes():
            self.params['network'].node[i]['state'] = self.params['initial_states'][count_nodes]
            if self.params['network'].node[i]['state'] == infected:
                for j in self.params['network'].neighbors(i):
                    self.params['network'].node[j]['number_of_infected_neighbors'] += 1
            count_nodes += 1

        self.number_of_infected_neighbors_is_updated = True

        assert self.number_of_infected_neighbors_is_updated and self.time_since_infection_is_updated, \
            'error: number_of_infected_neighbors or time_since_infection mishandle'

    def maslov_sneppen_rewiring(self, num_steps = SENTINEL, return_connected = True):
        """
        Rewire the network graph according to the Maslov and
        Sneppen method for degree-preserving random rewiring of a complex network,
        as described on
        `Maslov's webpage <http://www.cmth.bnl.gov/~maslov/matlab.htm>`_.
        Return the resulting graph.
        If a positive integer ``num_steps`` is given, then perform ``num_steps``
        number of steps of the method.
        Otherwise perform the default number of steps of the method, namely
        ``4*graph.num_edges()`` steps.
        The code is adopted from: https://github.com/araichev/graph_dynamics/blob/master/graph_dynamics.py
        """
        assert 'network' in self.params, 'error: network is not yet not set.'

        if num_steps is SENTINEL:
            num_steps = 10 * self.params['network'].number_of_edges()
            # completely rewire everything

        rewired_network = copy.deepcopy(self.params['network'])
        for i in range(num_steps):
            chosen_edges = RD.sample(rewired_network.edges(), 2)
            e1 = chosen_edges[0]
            e2 = chosen_edges[1]
            new_e1 = (e1[0], e2[1])
            new_e2 = (e2[0], e1[1])
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1] or \
                    rewired_network.has_edge(*new_e1) or rewired_network.has_edge(*new_e2):
                # Not allowed to rewire e1 and e2. Skip.
                continue

            rewired_network.remove_edge(*e1)
            rewired_network.remove_edge(*e2)
            rewired_network.add_edge(*new_e1)
            rewired_network.add_edge(*new_e2)

        if return_connected:
            while not NX.is_connected(rewired_network):
                rewired_network = copy.deepcopy(self.params['network'])
                for i in range(num_steps):
                    chosen_edges = RD.sample(rewired_network.edges(), 2)
                    e1 = chosen_edges[0]
                    e2 = chosen_edges[1]
                    new_e1 = (e1[0], e2[1])
                    new_e2 = (e2[0], e1[1])
                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1] or \
                            rewired_network.has_edge(*new_e1) or rewired_network.has_edge(*new_e2):
                        # Not allowed to rewire e1 and e2. Skip.
                        continue

                    rewired_network.remove_edge(*e1)
                    rewired_network.remove_edge(*e2)
                    rewired_network.add_edge(*new_e1)
                    rewired_network.add_edge(*new_e2)

        return rewired_network

    def random_random_rewiring(self, num_edges=SENTINEL, return_connected=True):
        """
        Rewire the network graph.
        Choose num_edges randomly from the existing edges and remove them.
        Choose num_edges randomly from the non-existing edges and add them.
        """
        assert 'network' in self.params, 'error: network is not yet not set.'

        if num_edges is SENTINEL:
            num_edges = self.params['network'].number_of_edges()
            print('Warning: number of edges to rewire not supplied, all edges will be rewired.')
            # completely rewire everything

        rewired_network = copy.deepcopy(self.params['network'])

        unformed_edges = list(NX.non_edges(rewired_network))

        formed_edges = list(NX.edges(rewired_network))

        addition_list = np.random.choice(range(len(unformed_edges)),
                                         num_edges,
                                         replace=False)

        addition_list = addition_list.astype(int)

        addition_list = [unformed_edges[ii] for ii in list(addition_list)]

        rewired_network.add_edges_from(addition_list)

        removal_list = np.random.choice(range(len(formed_edges)),
                                        num_edges,
                                        replace=False)

        removal_list = removal_list.astype(int)

        removal_list = [formed_edges[ii] for ii in list(removal_list)]

        rewired_network.add_edges_from(removal_list)

        if return_connected:
            while not NX.is_connected(rewired_network):
                rewired_network = copy.deepcopy(self.params['network'])

                unformed_edges = list(NX.non_edges(rewired_network))

                formed_edges = list(NX.edges(rewired_network))

                addition_list = np.random.choice(range(len(unformed_edges)),
                                                 num_edges,
                                                 replace=False)

                addition_list = addition_list.astype(int)

                addition_list = [unformed_edges[ii] for ii in list(addition_list)]

                rewired_network.add_edges_from(addition_list)

                removal_list = np.random.choice(range(len(formed_edges)),
                                                num_edges,
                                                replace=False)

                removal_list = removal_list.astype(int)

                removal_list = [formed_edges[ii] for ii in list(removal_list)]

                rewired_network.remove_edges_from(removal_list)

        return rewired_network

    def setRandomParams(self):
        """
        the parameters that are provided when the class is being initialized are treated as fixed. The missing parameters
        are set randomly. In an inference framework the distributions that determine the random draws are priors are those
        parameters which are not fixed.
        """
        assert self.missing_params_not_set,'error: missing parameters are already set.'
        # no spontaneous adoptions
        if 'zero_at_zero' not in self.fixed_params:
            self.params['zero_at_zero'] = True
        # below threshold adoption rate is divided by the self.params['multiplier']
        if 'multiplier' not in self.fixed_params:
            self.params['multiplier'] = 5
        # the high probability in complex contagion
        if 'fixed_prob_high' not in self.fixed_params:
            self.params['fixed_prob_high'] = 1.0
        # the low probability in complex contagion
        if 'fixed_prob' not in self.fixed_params:
            self.params['fixed_prob'] = 0.0
        # SI infection rate
        if 'beta' not in self.fixed_params:
            self.params['beta'] =  RD.choice([0.2,0.3,0.4,0.5])#0.1 * np.random.beta(1, 1, None)#0.2 * np.random.beta(1, 1, None)
        if 'sigma' not in self.fixed_params:
            self.params['sigma'] = RD.choice([0.1, 0.3, 0.5, 0.7,1])
        # complex contagion threshold
        if 'theta' not in self.fixed_params:
            self.params['theta'] = RD.choice([1, 2, 3, 4])#np.random.randint(1, 4)
        if 'size' not in self.fixed_params:
            if 'network' in self.fixed_params:
                self.params['size'] = NX.number_of_nodes(self.params['network'])
            else:
                self.params['size'] = 100  # np.random.randint(50, 500)
        if 'network_model' not in self.fixed_params:
            self.params['network_model'] = RD.choice(['erdos_renyi','watts_strogatz','grid','random_regular'])



        self.init_network()
        self.init_network_states()

        self.missing_params_not_set = False


class contagion_model(network_model):
    """
    implements data generation
    """
    def __init__(self,params):
        super(contagion_model,self).__init__()
        self.fixed_params = copy.deepcopy(params)
        self.params = params
        self.missing_params_not_set = True
        self.number_of_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False

    def time_the_total_spread(self, get_network_time_series = False, verbose = False):
        time = 0
        network_time_series = []

        self.missing_params_not_set = True
        self.setRandomParams()

        if hasattr(self, 'isActivationModel'):
            self.set_activation_functions()

        dummy_network = self.params['network'].copy()
        network_time_series.append(dummy_network)

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                self.params['network'].nodes()))
        total_number_of_infected = np.sum(all_nodes_states)
        if verbose:
            print('time is', time)
            print('total_number_of_infected is', total_number_of_infected)
            print('total size is', self.params['size'])
        while total_number_of_infected < self.params['size']:
            self.outer_step()
            dummy_network = self.params['network'].copy()
            network_time_series.append(dummy_network)
            time += 1
            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                    self.params['network'].nodes()))
            total_number_of_infected = np.sum(all_nodes_states)

            if verbose:
                print('time is', time)
                print('total_number_of_infected is', total_number_of_infected)
                print('total size is', self.params['size'])

            if time > self.params['size']**3:
                time = -1
                print('takes too long to spread totally.')
                break

        if get_network_time_series:
            return time, network_time_series
        else:
            return time

    def generateTimeseries(self):
        # conditioned on the fixed_params
        # returns the time series from fraction of infected nodes.
        total_time = self.time_the_total_spread()
        # print(total_time)

        time = 0
        self.missing_params_not_set = True
        self.setRandomParams()
        if hasattr(self, 'isActivationModel'):
            self.set_activation_functions()
        network_timeseries = []



        while time < total_time:
            dummy_network = self.params['network'].copy()
            network_timeseries.append(dummy_network)
            self.outer_step()
            time += 1

        all_nodes_states = list(map(lambda node_pointer:
                                    list(map(lambda network: 1.0*(network.node[node_pointer]['state']),
                                             network_timeseries)), self.params['network'].nodes()))

        fractions_timeseies = np.sum(all_nodes_states, 0)/self.params['size']

        df = pd.DataFrame(fractions_timeseies)
        # print(df)
        #df = pd.DataFrame(np.transpose(all_nodes_states))
        return df

    def genTorchSample(self):
        df = self.generateTimeseries()
        data = torch.FloatTensor(df.values[:, 0:1])  # [:,0:feature_length] pass on the fraction of infected people
        label_of_data = torch.LongTensor([self.classification_label])
        return label_of_data, data

    def genTorchDataset(self,dataset_size=1000):
        simulation_results = []
        for iter in range(dataset_size):
            label_of_data, data = self.genTorchSample()
            simulation_results.append((label_of_data, data))
        return simulation_results

    def avg_speed_of_spread(self , dataset_size=1000,cap=0.9, mode='max'):
        # avg time to spread over the dataset.
        # The time to spread is measured in one of the modes:
        # integral, max, and total.
        if mode == 'integral':
            dataset = self.genTorchDataset(dataset_size)
            integrals = []
            sum_of_integrals = 0
            for i in range(dataset_size):
                integral = sum(dataset[i][1][:,0])/len(dataset[i][1])
                sum_of_integrals += integral
                integrals += [integral]
            avg_speed = sum_of_integrals/dataset_size
            speed_std = np.std(integrals)
            speed_max = np.max(integrals)
            speed_min = np.min(integrals)
            speed_samples = np.asarray(integrals)

        elif mode == 'max':
            dataset = self.genTorchDataset(dataset_size)
            cap_times = []
            sum_of_cap_times = 0
            for i in range(dataset_size):
                cap_time = time_the_cap(dataset[i][1][:,0], cap)
                if cap_time == -1:
                    dataset_size += -1
                    continue
                sum_of_cap_times += cap_time
                cap_times += [cap_time]
            if dataset_size == 0:
                avg_speed = -1
            else:
                avg_speed = sum_of_cap_times/dataset_size
                speed_std = np.std(cap_times)
                speed_max = np.max(cap_times)
                speed_min = np.min(cap_times)
                speed_samples = np.asarray(cap_times)
        elif mode == 'total':
            total_spread_times = []
            sum_of_total_spread_times = 0
            count = 1
            while count <= dataset_size:
                total_spread_time = self.time_the_total_spread()
                # print('total_spread_time',total_spread_time)
                if total_spread_time == -1:
                    dataset_size += -1
                    print('The contagion did not spread totally.')
                    continue
                total_spread_times += [total_spread_time]
                sum_of_total_spread_times += total_spread_time
                count += 1
            if dataset_size == 0:
                avg_speed = -1
            else:
                avg_speed = sum_of_total_spread_times/dataset_size
                speed_std = np.std(total_spread_times)
                speed_max = np.max(total_spread_times)
                speed_min = np.min(total_spread_times)
                speed_samples = np.asarray(total_spread_times)

        else:
            assert False, 'undefined mode for avg_speed_of_spread'

        if type(avg_speed) is torch.Tensor:
            avg_speed = avg_speed.item()

        return avg_speed, speed_std, speed_max, speed_min, speed_samples

    def outer_step(self):
        assert hasattr(self, 'classification_label'), 'classification_label not set'
        assert not self.missing_params_not_set, 'missing params are not set'
        self.number_of_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.step()
        assert self.number_of_infected_neighbors_is_updated and self.time_since_infection_is_updated, \
            'error: number_of_infected_neighbors or time_since_infection mishandle'

    def step(self):
        pass


class activation(contagion_model):
    """
    allows for the probability of infection to be set dependent on the number of infected neighbors, according to
    an activation function.
    """
    def __init__(self,params,externally_set_activation_functions=SENTINEL,externally_set_classification_label = SENTINEL):

        super(activation, self).__init__(params)

        assert not (
        (externally_set_activation_functions == SENTINEL and externally_set_classification_label != SENTINEL) or
        (externally_set_activation_functions != SENTINEL and externally_set_classification_label == SENTINEL)), \
            'externally_set_activation_function and classification_label should be provided together'

        self.externally_set_activation_functions = externally_set_activation_functions
        self.externally_set_classification_label = externally_set_classification_label

        self.activation_functions = []
        self.activation_functions_is_set = False

        print(self.params)

        if 'size' not in self.params:
            if 'network' in self.params:
                self.params['size'] = NX.number_of_nodes(self.params['network'])
            else:
                assert False, 'neither network size nor size are defined'

        assert 'size' in self.params.keys(), 'the network size should be set at this stage'
        self.activation_probabilities = list(range(self.params['size']))
        self.activation_probabilities_is_set = False

        self.isActivationModel = True

    def set_activation_functions(self):
        """
        This will be called prior to generating each timeseries if self.isActivationModel
        It sets the activation functions for the nodes
        Each node's activation function can be accessed as follows:
        self.activation_functions[index of the node](number of infected neighbors)
        This function is overwritten by each of the subclasses of activation() such as probit, logit, etc
        """
        if self.externally_set_activation_function != SENTINEL:
            assert self.externally_set_classification_label != SENTINEL, 'classification_label not provided'
            for node_counter in range(self.params['size']):
                self.activation_functions.append(lambda number_of_infected_neighbors , i=node_counter :
                                                 self.externally_set_activation_function[i][number_of_infected_neighbors])
            self.activation_functions_is_set = True
        else:
            assert self.externally_set_classification_label == SENTINEL, \
                'classification_label provided without externally_set_activation_functions'
            self.activation_functions_is_set = False

    def set_activation_probabilities(self):
        """
        This will be called at every time step during the generation of a time series to update the activation probabilities based on the number of infected neighbors
        """
        assert self.activation_functions_is_set, 'activation_fucntions are not set'

        current_network = copy.deepcopy(self.params['network'])
        count_nodes = 0
        for i in current_network.nodes():
            self.activation_probabilities[count_nodes] = \
                self.activation_functions[count_nodes](current_network.node[i]['number_of_infected_neighbors'])
            count_nodes += 1
        self.activation_probabilities_is_set = True

    def step(self):

        self.set_activation_probabilities()

        assert self.activation_probabilities_is_set, 'activation_probabilities not set'

        current_network = copy.deepcopy(self.params['network'])
        count_node = 0

        for i in current_network.nodes():
            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0, \
                    'error: time_since_infection'
                self.time_since_infection_is_updated = True
                if RD.random() < self.activation_probabilities[count_node]:
                    self.params['network'].node[i]['state'] = infected
                    for k in self.params['network'].neighbors(i):
                        self.params['network'].node[k]['number_of_infected_neighbors'] += 1
                self.number_of_infected_neighbors_is_updated = True

            elif RD.random() < self.params['delta']:
                self.params['network'].node[i]['state'] = susceptible
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_infected_neighbors'] -= 1
                self.number_of_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.time_since_infection_is_updated = True
            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                self.time_since_infection_is_updated = True
                self.number_of_infected_neighbors_is_updated = True
            count_node += 1

    def get_activation_probabilities(self,degree_range=range(10),node = SENTINEL):
        if self.missing_params_not_set:
            self.setRandomParams()
        if not self.activation_functions_is_set:
            self.set_activation_functions()
        activation_probabilities = []
        if node == SENTINEL:
            node = RD.choice(list(range(self.params['size'])))
        for i in degree_range:
            activation_probabilities.append(self.activation_functions[node](i))
        return activation_probabilities


class SIS(activation):
    """
    SIS spread model infection probability beta, recovery probability delta
    """
    def __init__(self,params):
        super(SIS,self).__init__(params)
        self.classification_label = SIMPLE

    def set_activation_functions(self):
        """
        sets the SIS activation functions for each of the nodes
        """
        self.activation_functions = \
            [lambda number_of_infected_neighbors:1 - (1 - self.params['beta'])**number_of_infected_neighbors]\
            *self.params['size']
        self.activation_functions_is_set = True


class SIS_threshold(activation):
    """
    threshold SIS model, threshold is theta if all nodes have the same thershold. If number of infected neighbors is
    strictly greater than theta then the node would get infected with independent infection probability beta (SIS).
    Below the threshold the nodes gets infected with a fixed probability (fixed_prob) as long as it has at least one
    infected neighbor.
    """
    def __init__(self,params):
        super(SIS_threshold, self).__init__(params)
        self.classification_label = COMPLEX

    def set_activation_functions(self):
        """
        sets the SIS_threshold activation functions for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        if self.params['zero_at_zero']:
            for i in current_network.nodes():
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                 ((1 - (1 - self.params['beta']) ** number_of_infected_neighbors) * 1.0 * (
                                                     current_network.node[i]['threshold'] < number_of_infected_neighbors)) +
                                                 (self.params['fixed_prob']* 1.0 *
                                                 ((current_network.node[i]['threshold'] >= number_of_infected_neighbors)
                                                  and (not (number_of_infected_neighbors < 1)))) +
                                                 0.0 * (number_of_infected_neighbors < 0))
        else:  # not zero at zero
            for i in current_network.nodes():
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                 ((1 - (1 - self.params['beta']) ** number_of_infected_neighbors) * 1.0 * (
                                                     current_network.node[i]['threshold'] < number_of_infected_neighbors)) +
                                                 (self.params['fixed_prob'] * 1.0 *
                                                  (current_network.node[i]['threshold'] >=
                                                   number_of_infected_neighbors)))

        self.activation_functions_is_set = True


class SIS_threshold_soft(activation):
    """
    Similar to threshold SIS model, expect that below the threshold, nodes follow SIS with a decreased infection
    probability (beta/multiplier).
    """
    def __init__(self,params):
        super(SIS_threshold_soft, self).__init__(params)
        self.classification_label = COMPLEX

    def set_activation_functions(self):
        """
        sets the SIS_threshold_soft activation functions for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        for i in current_network.nodes():
            self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                             (1 - (1 - self.params['beta']) ** number_of_infected_neighbors) * 1.0 * (
                                                 current_network.node[i]['threshold'] < number_of_infected_neighbors) +
                                             (1 - ((1 - (self.params['beta']/self.params['multiplier']))
                                                   ** number_of_infected_neighbors))
                                             * 1.0 * (current_network.node[i]['threshold'] >=
                                                      number_of_infected_neighbors))
            # this is always zero at zero.
        self.activation_functions_is_set = True


class Probit(activation):
    """
    Probit activation function. Probability of adoption is set according to a normal CDF with mean theta and
    scale factor sigma.
    """
    def __init__(self,params):
        super(Probit, self).__init__(params)
        self.classification_label = COMPLEX

    def set_activation_functions(self):
        """
        sets the probit activation functions for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        thresholds = NX.get_node_attributes(current_network, 'threshold')
        if len(set(thresholds.values())) == 1:  # all nodes have the same threshold
            probit_function = norm(self.params['theta'], self.params['sigma'])
            if self.params['zero_at_zero']:
                self.activation_functions = \
                    [lambda number_of_infected_neighbors:
                     1.0 * (number_of_infected_neighbors > 0) * probit_function.cdf(number_of_infected_neighbors) +
                     0.0 * 1.0 * (number_of_infected_neighbors == 0)]*self.params['size']
            else:  # not zero at zero:
                self.activation_functions = \
                    [lambda number_of_infected_neighbors:
                     1.0 * probit_function.cdf(number_of_infected_neighbors)] * self.params['size']

        else:  # heterogeneous thresholds
            if self.params['zero_at_zero']:

                for i in current_network.nodes():
                    probit_function = norm(current_network.node[i]['threshold'], self.params['sigma'])
                    if self.params['zero_at_zero']:
                        self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                         (1.0 * (number_of_infected_neighbors > 0)
                                                          * probit_function.cdf(number_of_infected_neighbors)) +
                                                         0.0 * 1.0 * (number_of_infected_neighbors == 0))
                    else:  # not zer_zt_zero
                        self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                         (1.0 * probit_function.cdf(number_of_infected_neighbors)))

        self.activation_functions_is_set = True


class Logit(activation):
    """
    Logit activation function with threshold theta and scale sigma.
    """
    def __init__(self,params):
        super(Logit, self).__init__(params)
        self.classification_label = COMPLEX

    def set_activation_functions(self):
        """
        sets the logit activation functions for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        thresholds = NX.get_node_attributes(current_network, 'threshold')
        if len(set(thresholds.values())) == 1:  # all nodes have the same threshold
            if self.params['zero_at_zero']:
                self.activation_functions = \
                    [lambda number_of_infected_neighbors: ((1/(1 + np.exp((1 / self.params['sigma']) * (
                            self.params['theta'] - number_of_infected_neighbors))))
                                                           * 1.0 * (number_of_infected_neighbors > 0)) +
                                                          0.0 * 1.0 * (number_of_infected_neighbors == 0)
                     ]*self.params['size']
            else:  # not zero_at_zero
                self.activation_functions = \
                    [lambda number_of_infected_neighbors: (1 / (1 + np.exp((1 / self.params['sigma']) * (
                            self.params['theta'] - number_of_infected_neighbors))))* 1.0] * self.params['size']
        else:  # heterogeneous thresholds
            if self.params['zero_at_zer0']:
                for i in current_network.nodes():
                    self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i :
                                                     ((1/(1 + np.exp((1 / self.params['sigma']) *
                                                            (current_network.node[node_index]['threshold'] -
                                                             number_of_infected_neighbors)))) *
                                                      1.0 * (number_of_infected_neighbors > 0)) +
                                                     0.0 * 1.0 * (number_of_infected_neighbors == 0))
            else:  # not zero at zero
                for i in current_network.nodes():
                    self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i :
                                                     (1/(1 + np.exp((1 / self.params['sigma']) *
                                                            (current_network.node[node_index]['threshold'] -
                                                             number_of_infected_neighbors)))) * 1.0)
        self.activation_functions_is_set = True


class Linear(activation):
    """
    Implements a linear threshold activation function. Thresholds are set relative to the total number of neighbors
    (ratios of infected in the local neighborhoods). The threshold values are set uniformly at random from the (0,1)
    interval. Above the threshold infection happens with fixed probability 'fixed_prob_high'. Below the threshold
    infection happens with fixed probability 'fixed_prob'. The fixed probabilities can be set to one or zero.
    """
    def __init__(self,params):
        super(Linear, self).__init__(params)
        self.classification_label = COMPLEX
        self.isLinearThresholdModel = True

    def set_activation_functions(self):
        """
        sets the linear threshold activation functions for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        for i in current_network.nodes():  # thresholds are heterogeneous by definition of the linear threshold model
            # they are set random uniform (0,1) ratios.
            if self.params['zero_at_zero']:
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i :
                                                 self.params['fixed_prob_high'] * 1.0 *
                                                 (current_network.node[i]['threshold'] <= number_of_infected_neighbors)
                                                 + self.params['fixed_prob'] * 1.0 * (
                                                     (current_network.node[i]['threshold'] >
                                                      number_of_infected_neighbors)
                                                     and (number_of_infected_neighbors != 0)) +
                                                 0.0 * 1.0 * (number_of_infected_neighbors == 0))
            else: # not zero_at_zero
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                 self.params['fixed_prob_high'] * 1.0 *
                                                 (current_network.node[i]['threshold'] <= number_of_infected_neighbors)
                                                 + self.params['fixed_prob'] * 1.0 * (
                                                         (current_network.node[i]['threshold'] >
                                                          number_of_infected_neighbors)))

        self.activation_functions_is_set = True


class DeterministicLinear(activation):
    """
    Similar to the  linear threshold model except that thresholds are not ratios and are not random. 'thresholds' are
    fixed and they can be set all the same equal to theta.
    """
    def __init__(self, params):
        super(DeterministicLinear, self).__init__(params)
        self.classification_label = COMPLEX

    def set_activation_functions(self):
        """
        sets the linear threshold activation functions with deterministic thresholds for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        for i in current_network.nodes():
            if self.params['zero_at_zero']:
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                 self.params['fixed_prob_high'] * 1.0 * (
                                                     current_network.node[i][
                                                         'threshold'] <= number_of_infected_neighbors) +
                                                 self.params['fixed_prob'] * 1.0 * (
                                                         (current_network.node[i][
                                                              'threshold'] > number_of_infected_neighbors)
                                                         and (not (number_of_infected_neighbors < 1))) +
                                                 0.0 * 1.0 * (number_of_infected_neighbors < 1))
            else:  # not zero at zero
                self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                 self.params['fixed_prob_high'] * 1.0 * (
                                                         current_network.node[i][
                                                             'threshold'] <= number_of_infected_neighbors) +
                                                 self.params['fixed_prob'] * 1.0 * ((current_network.node[i][
                                                              'threshold'] > number_of_infected_neighbors)))

        self.activation_functions_is_set = True



class SimpleOnlyAlongC1(contagion_model):
    """
    Implements an a special contagion model that is useful for interpolating C_1 and C_2 when studying the effect
    of rewiring. It allows complex contagion along all edges. Also allows simple contagion
    but only along cycle edges.
    """

    def __init__(self, params):
        super(SimpleOnlyAlongC1, self).__init__(params)
        self.classification_label = COMPLEX
        assert self.params['network_model'] == 'c_1_c_2_interpolation', \
            "this contagion model is only suitable for c_1_c_2_interpolation"

    def step(self):
        current_network = copy.deepcopy(self.params['network'])
        for i in current_network.nodes():
            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0, \
                    'error: time_since_infection mishandle'
                self.time_since_infection_is_updated = True
                # first check if the node can be infected through complex contagion
                # print('node i', i)
                # print('current_network.node[i][threshold]', current_network.node[i]['threshold'])
                # print('.node[i][number_of_infected_neighbors',
                #       self.params['network'].node[i]['number_of_infected_neighbors'])
                if (current_network.node[i]['threshold'] <=
                        current_network.node[i]['number_of_infected_neighbors']):
                    # print('we are here')
                    if RD.random() < self.params['fixed_prob_high']:
                        self.params['network'].node[i]['state'] = infected
                        for k in self.params['network'].neighbors(i):
                            self.params['network'].node[k]['number_of_infected_neighbors'] += 1

                else:  # if the node cannot be infected through complex contagion
                    # see if it can be infected through simple contagion along cycle edges
                    for j in current_network.neighbors(i):
                        # print('i', i)
                        # print('j', j)
                        j_is_a_cycle_neighbor = ((abs(i - j) == 1) or
                                                 (abs(i - j) == (self.params['size'] - 1)))
                        # print('j_is_a_cycle_neighbor', j_is_a_cycle_neighbor)
                        if (current_network.node[j]['state'] == infected
                                and j_is_a_cycle_neighbor):
                            if RD.random() < self.params['fixed_prob']:
                                self.params['network'].node[i]['state'] = infected
                                for k in self.params['network'].neighbors(i):
                                    self.params['network'].node[k]['number_of_infected_neighbors'] += 1
                                break

                self.number_of_infected_neighbors_is_updated = True # the updating should happen in
                # self.params['network'] not the current network

            # if node i is already infected but recovers
            elif RD.random() < self.params['delta']:
                    self.params['network'].node[i]['state'] = susceptible
                    for k in self.params['network'].neighbors(i):
                        self.params['network'].node[k]['number_of_infected_neighbors'] -= 1
                    self.number_of_infected_neighbors_is_updated = True
                    self.params['network'].node[i]['time_since_infection'] = 0
                    self.time_since_infection_is_updated = True
            # if node i is already infected and does not recover
            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                self.time_since_infection_is_updated = True
                self.number_of_infected_neighbors_is_updated = True

class IndependentCascade(contagion_model):
    """
    Implements an independent cascade model. Each infected neighbor has an independent probability beta of passing on her
    infection, as long as her infection has occurred within the past mem = 1 time steps.
    """
    def __init__(self,params,mem = 1):
        super(IndependentCascade, self).__init__(params)
        self.classification_label = SIMPLE
        self.memory = mem

    def step(self):
        current_network = copy.deepcopy(self.params['network'])
        for i in current_network.nodes():
            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0, \
                    'error: time_since_infection mishandle'
                self.time_since_infection_is_updated = True
                for j in current_network.neighbors(i):
                    if (current_network.node[j]['state'] == infected and
                            current_network.node[j]['time_since_infection'] < self.memory):
                        if RD.random() < self.params['beta']:
                            self.params['network'].node[i]['state'] = infected
                            for k in self.params['network'].neighbors(i):
                                self.params['network'].node[k]['number_of_infected_neighbors'] += 1
                            break
                self.number_of_infected_neighbors_is_updated = True

            elif RD.random() < self.params['delta']:
                    self.params['network'].node[i]['state'] = susceptible
                    for k in self.params['network'].neighbors(i):
                        self.params['network'].node[k]['number_of_infected_neighbors'] -= 1
                    self.number_of_infected_neighbors_is_updated = True
                    self.params['network'].node[i]['time_since_infection'] = 0
                    self.time_since_infection_is_updated = True
            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                self.time_since_infection_is_updated = True
                self.number_of_infected_neighbors_is_updated = True


class NewModel(contagion_model):
    def __init__(self,params):
        super(NewModel, self).__init__(params)
        self.classification_label = SIMPLE # or CMOPLEPX
        # set other model specific flags and handles here

    def set_activation_functions(self):
        """
        sets the linear threshold activation functions with deterministic thresholds for each of the nodes
        """
        current_network = copy.deepcopy(self.params['network'])

        for i in current_network.nodes():
            if self.params['zero_at_zero']:
                pass
            else:  # not zero at zero
                pass
        self.activation_functions_is_set = True

    # note: do not implement step if you implemented set_activation_functions()

    def step(self):
        current_network = copy.deepcopy(self.params['network'])
        for i in current_network.nodes():
            # set node states according to the model
            pass
        # makes sure time_since_infection and number_of_infected_neighbors are properly updated
        self.time_since_infection_is_updated = True
        self.number_of_infected_neighbors_is_updated = True