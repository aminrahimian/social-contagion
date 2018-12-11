from settings import *

import copy

from scipy.stats import norm


def measure_property(network_intervention_dataset, property='avg_clustering', sample_size=None):

    if sample_size is not None:
        assert sample_size <= len(network_intervention_dataset), \
            "not enough samples to do measurements on network_intervention_dataset, sample_size: " + str(sample_size) \
            + "len(network_intervention_dataset): " + str(len(network_intervention_dataset))
        network_intervention_dataset = network_intervention_dataset[0:sample_size-1]

    property_samples = []

    for network_intervention in network_intervention_dataset:
        if property is 'avg_clustering':
            property_sample = NX.average_clustering(network_intervention)
        elif property is 'average_shortest_path_length':
            property_sample = NX.average_shortest_path_length(network_intervention)
        elif property is 'diameter':
            property_sample = NX.diameter(network_intervention)
        elif property is 'size_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            property_sample = sample_2_core.number_of_nodes()
        elif property is 'avg_degree':
            degree_sequence = [d for n, d in network_intervention.degree()]
            sum_of_edges = sum(degree_sequence)
            property_sample = sum_of_edges/network_intervention.number_of_nodes()
        elif property is 'diam_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('Inf')
            else:
                property_sample = NX.diameter(sample_2_core)
        elif property is 'max_degree':
            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = max(degree_sequence)
        elif property is 'min_degree':
            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = min(degree_sequence)
        elif property is 'max_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = sorted([d for n, d in sample_2_core.degree()], reverse=True)
                property_sample = max(degree_sequence)
        elif property is 'min_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = sorted([d for n, d in sample_2_core.degree()], reverse=True)
                property_sample = min(degree_sequence)
        elif property is 'avg_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = [d for n, d in sample_2_core.degree()]
                sum_of_edges = sum(degree_sequence)
                property_sample = sum_of_edges / sample_2_core.number_of_nodes()
        elif property is 'number_edges':
            property_sample = network_intervention.number_of_edges()
        elif property is 'number_edges_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                property_sample = sample_2_core.number_of_edges()
        elif property is 'avg_clustering_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('NaN')
            else:
                property_sample = NX.average_clustering(sample_2_core)
        elif property is 'transitivity':
            property_sample = NX.transitivity(network_intervention)
        elif property is 'transitivity_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('NaN')
            else:
                property_sample = NX.transitivity(sample_2_core)
        elif property is 'num_leaves':

            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = degree_sequence.count(int(1))
        else:
            assert False, property + ' property not supported.'

        property_samples += [property_sample]

    return property_samples


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


def newman_watts_add_fixed_number_graph(n, k=2, p=2, seed=None):
    """Returns a Newman–Watts–Strogatz small-world graph. With a fixed - p - (not random) number of edges
    added to each node. Modified newman_watts_strogatzr_graph() in NetworkX.
    """
    if seed is not None:
        RD.seed(seed)
    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    G = NX.connected_watts_strogatz_graph(n, k, 0)

    all_nodes = G.nodes()
    for u in all_nodes:
        count_added_edges = 0  # track number of edges added to node u
        while count_added_edges < p:

            w = np.random.choice(all_nodes)

            # no self-loops and reject if edge u-w exists
            # is that the correct NWS model?
            while w == u or G.has_edge(u, w):
                w = np.random.choice(all_nodes)
                # print('re-drawn w', w)
                if G.degree(u) >= n-1:
                    break # skip this rewiring
            G.add_edge(u, w)
            count_added_edges += 1
    return G


def cycle_union_Erdos_Renyi(n, k=4, c=2, seed=None,
                            color_the_edges=False,
                            cycle_edge_color='k',
                            random_edge_color='b',
                            weight_the_edges=False,
                            cycle_edge_weights=4,
                            random_edge_weights=4):
    """Returns a cycle C_k union G(n,c/n) graph by composing
    NX.connected_watts_strogatz_graph(n, k, 0) and
    NX.erdos_renyi_graph(n, c/n, seed=None, directed=False)"""
    if seed is not None:
        RD.seed(seed)

    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    C_k = NX.connected_watts_strogatz_graph(n, k, 0)
    if color_the_edges:
        # cycle_edge_colors = dict.fromkeys(C_k.edges(), cycle_edge_color)
        NX.set_edge_attributes(C_k, cycle_edge_color, 'color')
    if weight_the_edges:
        NX.set_edge_attributes(C_k, cycle_edge_weights, 'weight')

    G_npn = NX.erdos_renyi_graph(n, c/n, seed=None, directed=False)
    if color_the_edges:
        # random_edge_colors = dict.fromkeys(G_npn.edges(), random_edge_color)
        NX.set_edge_attributes(G_npn, random_edge_color, 'color')
    if weight_the_edges:
        NX.set_edge_attributes(G_npn, random_edge_weights, 'weight')

    assert G_npn.nodes() == C_k.nodes(), "node sets are not the same"
    composed = NX.compose(G_npn, C_k)

    # print(composed.edges.data())

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

    for edge in C_2.edges():
        # print(edge)
        if abs(edge[0] - edge[1]) == 2 or abs(edge[0] - edge[1]) == n-2: # it is a C_2\C_1 edge
            if remove_cycle_edges_exp[C_2_minus_C_1_edge_index] < eta:
                removal_list += [edge]
            C_2_minus_C_1_edge_index += 1 # index the next C_2\C_1 edge
    C_2.remove_edges_from(removal_list)

    addition_list = []

    K_n = NX.complete_graph(n)

    random_add_edge_index = 0

    for edge in K_n.edges():

        if add_long_ties_exp[random_add_edge_index] < eta:
            addition_list += [edge]
        random_add_edge_index += 1 # index the next edge to be considered for addition
    C_2.add_edges_from(addition_list)

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

    unformed_edges = list(NX.non_edges(fat_network))

    if len(unformed_edges) < number_of_edges_to_be_added:
        print("There are not that many edges left ot be added")
        fat_network.add_edges_from(unformed_edges)  # add all the edges that are left
        return fat_network

    if mode is 'random':
        addition_list = RD.sample(unformed_edges, number_of_edges_to_be_added)
        fat_network.add_edges_from(addition_list)
        return fat_network

    if mode is 'triadic_closures':
        weights = []
        for non_edge in unformed_edges:
            weights += [len(list(NX.common_neighbors(G, non_edge[0], non_edge[1])))]

        total_sum = sum(weights)

        normalized_weights = [weight/total_sum for weight in weights]

        addition_list = np.random.choice(range(len(unformed_edges)),
                                         number_of_edges_to_be_added,
                                         replace=False,
                                         p=normalized_weights)

        addition_list = addition_list.astype(int)

        addition_list = [unformed_edges[ii] for ii in list(addition_list)]

        fat_network.add_edges_from(addition_list)

        return fat_network


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
                self.params['network'] = NX.connected_watts_strogatz_graph(self.params['size'],
                                                                           self.params['nearest_neighbors'],
                                                                           self.params['rewiring_probability'])
            elif self.params['network_model'] == 'grid':
                if 'number_grid_rows' not in self.fixed_params:
                    if 'number_grid_columns' not in self.fixed_params:
                        (self.params['number_grid_columns'],self.params['number_grid_rows']) = \
                            random_factor_pair(self.params['size'])
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
                self.params['network'] = newman_watts_add_fixed_number_graph(self.params['size'],
                                                                             self.params['nearest_neighbors'],
                                                                             self.params['fixed_number_edges_added'])
            elif self.params['network_model'] == 'cycle_union_Erdos_Renyi':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = cycle_union_Erdos_Renyi(self.params['size'], self.params['nearest_neighbors'],
                                                                 self.params['c'])

            elif self.params['network_model'] == 'c_1_c_2_interpolation':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                if 'add_long_ties_exp' not in self.fixed_params:
                    self.params['add_long_ties_exp'] = np.random.exponential(scale=self.params['size'] ** 2,
                                                                             size=int(1.0 * self.params['size']
                                                                                      * (self.params['size'] - 1)) // 2)

                    self.params['remove_cycle_edges_exp'] = np.random.exponential(scale=2 * self.params['size'],
                                                                                  size=self.params['size'])

                self.params['network'] = c_1_c_2_interpolation(self.params['size'],self.params['eta'],
                                                               self.params['add_long_ties_exp'],
                                                               self.params['remove_cycle_edges_exp'])
            else:
                assert False, 'undefined network type'

        # when considering real network and interventions on them we may need to record the original network.
        # This is currently only used in SimpleOnlyAlongOriginalEdges(contagion_model)

        if 'original_network' in self.fixed_params:
            self.params['original_network'] = self.fixed_params['original_network']
        else:
            self.params['original_network'] = None

        # additional modifications / structural interventions to the network topology which include rewiring
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

            self.params['network'] = rewired_network

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

            self.params['network'] = fattened_network

        self.node_list = list(self.params['network'])  # used for indexing nodes in cases where
        # node attributes are available in a list. A typical application is as follows: self.node_list.index(i)
        # for i in self.params['network'].nodes():

    def init_network_states(self):
        """
        initializes the node states (infected/susceptible) and other node attributes such as number of infected neighbors
        and time since infection
        """

        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False
        self.list_of_most_recent_switches = [] # list of most recent switches is useful for speeding up pure (0,1)
        # complex contagion computations by shoteninig the loops. It won't be updated unless we have a pure (0,1)
        # complex contagion in the LinearThreshold model.
        self.list_of_most_recent_switches_is_updated = False

        if 'initial_states' in self.fixed_params:
            for i in range(NX.number_of_nodes(self.params['network'])):
                if self.params['initial_states'][i] not in [susceptible, infected*active, infected*inactive]:
                    self.params['initial_states'][i] = infected*active
                    print('warning: states put to 0.5 for infected*active')

        elif 'initial_states' not in self.fixed_params:
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
                self.params['initial_states'][initially_infected_node_indexes] = infected * active
                # all nodes are initially active by default
                self.params['initial_states'] = list(self.params['initial_states'])

            else:
                assert False, "undefined initialization_mode"

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['number_of_active_infected_neighbors'] = 0
            self.params['network'].node[i]['time_since_infection'] = 0
            self.params['network'].node[i]['time_since_activation'] = 0
            self.params['network'].node[i]['threshold'] = self.params['thresholds'][self.node_list.index(i)]

        self.time_since_infection_is_updated = True
        self.time_since_activation_is_updated = True

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['state'] = self.params['initial_states'][self.node_list.index(i)]
            if self.params['network'].node[i]['state'] == infected * active:
                for j in self.params['network'].neighbors(i):
                    self.params['network'].node[j]['number_of_active_infected_neighbors'] += 1
                self.list_of_most_recent_switches.append(i)

        self.number_of_active_infected_neighbors_is_updated = True
        self.list_of_most_recent_switches_is_updated = True

        assert self.number_of_active_infected_neighbors_is_updated and self.time_since_infection_is_updated and \
               self.time_since_activation_is_updated and self.list_of_most_recent_switches_is_updated, \
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

        rewired_network.remove_edges_from(removal_list)

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
        assert self.missing_params_not_set, 'error: missing parameters are already set.'
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
        if 'beta' not in self.fixed_params:  # SIS infection rate parameter
            self.params['beta'] =  RD.choice([0.2,0.3,0.4,0.5])  # 0.1 * np.random.beta(1, 1, None)#0.2 * np.random.beta(1, 1, None)
        if 'sigma' not in self.fixed_params:  # logit and probit parameter
            self.params['sigma'] = RD.choice([0.1, 0.3, 0.5, 0.7,1])
        # complex contagion threshold
        if 'theta' not in self.fixed_params:  # complex contagion threshold parameter
            self.params['theta'] = RD.choice([1, 2, 3, 4])  # np.random.randint(1, 4)
        #  The default values gamma = 0 and alpha = 1 ensure that all infected nodes always remain active
        if 'gamma' not in self.fixed_params:  # rate of transition from active to inactive
            self.params['gamma'] = 0.0  # RD.choice([0.2,0.3,0.4,0.5])
        if 'alpha' not in self.fixed_params:  # rate of transition from inactive to active
            self.params['alpha'] = 1.0  # RD.choice([0.02,0.03,0.04,0.05])
        if 'size' not in self.fixed_params:
            if 'network' in self.fixed_params:
                self.params['size'] = NX.number_of_nodes(self.params['network'])
            else:
                self.params['size'] = 100  # np.random.randint(50, 500)
        if 'network_model' not in self.fixed_params:
            self.params['network_model'] = RD.choice(['erdos_renyi', 'watts_strogatz', 'grid', 'random_regular'])

        if 'thresholds' not in self.params:
            assert not hasattr(self, 'isLinearThresholdModel'), \
                "Thresholds should have been already set in the linear threshold model!"
            if 'thresholds' in self.fixed_params:
                self.params['thresholds'] = self.fixed_params['thresholds']
            else:
                self.params['thresholds'] = [self.params['theta']] * self.params['size']

        self.init_network()
        self.init_network_states()

        self.missing_params_not_set = False


class contagion_model(network_model):
    """
    implements data generation
    """
    def __init__(self, params):
        super(contagion_model, self).__init__()
        self.fixed_params = copy.deepcopy(params)
        self.params = params
        self.missing_params_not_set = True
        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False

    def time_the_total_spread(self, cap=0.99,
                              get_time_series=False,
                              verbose=False):
        time = 0
        network_time_series = []
        fractions_time_series = []

        self.missing_params_not_set = True
        self.setRandomParams()

        if hasattr(self, 'isActivationModel'):
            self.set_activation_functions()

        # record the values at time zero:
        dummy_network = self.params['network'].copy()

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                self.params['network'].nodes()))
        total_number_of_infected = 2*np.sum(abs(np.asarray(all_nodes_states)))
        fraction_of_infected = total_number_of_infected / self.params['size']

        if get_time_series:
            network_time_series.append(dummy_network)
            fractions_time_series.append(fraction_of_infected)

        if verbose:
            print('time is', time)
            print('total_number_of_infected is', total_number_of_infected)
            print('total size is', self.params['size'])
        while total_number_of_infected < cap*self.params['size']:
            self.outer_step()
            dummy_network = self.params['network'].copy()
            time += 1
            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                    self.params['network'].nodes()))
            total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
            fraction_of_infected = total_number_of_infected / self.params['size']
            if get_time_series:
                network_time_series.append(dummy_network)
                fractions_time_series.append(fraction_of_infected)
            if verbose:
                print('time is', time)
                print('total_number_of_infected is', total_number_of_infected)
                print('total size is', self.params['size'])
            if time > self.params['size']*10:
                time = float('Inf')
                print('It is taking too long (10x size) to spread totally.')
                break

        if get_time_series:
            return time, network_time_series, fractions_time_series
        else:
            return time

    def generate_network_intervention_dataset(self, dataset_size=200):

        interventioned_networks = []

        for i in range(dataset_size):
            self.missing_params_not_set = True
            self.setRandomParams()
            interventioned_networks += [self.params['network']]

        return interventioned_networks

    def avg_speed_of_spread(self, dataset_size=1000, cap=0.9, mode='max'):
        # avg time to spread over the dataset.
        # The time to spread is measured in one of the modes:
        # integral, max, and total.
        if mode == 'integral':
            integrals = []
            sum_of_integrals = 0
            for i in range(dataset_size):
                _, _, infected_fraction_timeseries = self.time_the_total_spread(cap=cap, get_time_series=True)
                integral = sum(infected_fraction_timeseries)
                sum_of_integrals += integral
                integrals += [integral]
            avg_speed = sum_of_integrals/dataset_size
            speed_std = np.std(integrals)
            speed_max = np.max(integrals)
            speed_min = np.min(integrals)
            speed_samples = np.asarray(integrals)
        elif mode == 'max':
            cap_times = []
            sum_of_cap_times = 0
            for i in range(dataset_size):
                time = self.time_the_total_spread(cap=cap, get_time_series=False)
                cap_time = time
                if cap_time == float('Inf'):
                    dataset_size += -1
                    cap_times += [float('Inf')]
                    continue
                sum_of_cap_times += cap_time
                cap_times += [cap_time]
            if dataset_size == 0:
                avg_speed = float('Inf')
                speed_std = float('NaN')
                speed_max = float('Inf')
                speed_min = float('Inf')
                speed_samples = np.asarray([float('Inf')])
            else:
                avg_speed = sum_of_cap_times/dataset_size
                speed_std = np.ma.std(cap_times)
                speed_max = np.max(cap_times)
                speed_min = np.min(cap_times)
                speed_samples = np.asarray(cap_times)
        elif mode == 'total':
            total_spread_times = []
            sum_of_total_spread_times = 0
            count = 1
            while count <= dataset_size:
                total_spread_time = self.time_the_total_spread(cap=0.999, get_time_series=False)
                if total_spread_time == float('Inf'):
                    dataset_size += -1
                    total_spread_times += [float('Inf')]
                    print('The contagion did not spread totally.')
                    continue
                total_spread_times += [total_spread_time]
                sum_of_total_spread_times += total_spread_time
                count += 1

            if dataset_size == 0:
                avg_speed = float('Inf')
                speed_std = float('NaN')
                speed_max = float('Inf')
                speed_min = float('Inf')
                speed_samples = np.asarray([float('Inf')])

            else:
                avg_speed = sum_of_total_spread_times/dataset_size
                speed_std = np.std(total_spread_times)
                speed_max = np.max(total_spread_times)
                speed_min = np.min(total_spread_times)
                speed_samples = np.asarray(total_spread_times)

        else:
            assert False, 'undefined mode for avg_speed_of_spread'

        return avg_speed, speed_std, speed_max, speed_min, speed_samples

    def outer_step(self):
        assert hasattr(self, 'classification_label'), 'classification_label not set'
        assert not self.missing_params_not_set, 'missing params are not set'
        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False
        self.step()
        # print(self.params['network'].edges())
        # print(self.params['network_model'])
        assert self.number_of_active_infected_neighbors_is_updated and self.time_since_infection_is_updated \
               and self.time_since_activation_is_updated, \
            'error: number_of_infected_neighbors or time_since_infection mishandle'

    def step(self):
        # implement this is activation class children
        pass


class activation(contagion_model):
    """
    Allows for the probability of infection to be set dependent on the number of infected neighbors, according to
    an activation function. Two main methods implemented here are set_activation_probabilities(self) and step(self)
    step is the most important method implemented in activation(contagion_model). It is used by all models that inherit
    activation There models that implement their own step functions (not inheriting activation) are
    SimpleOnlyAlongOriginalEdges(contagion_model), SimpleOnlyAlongC1(contagion_model),
    IndependentCascade(contagion_model)
    """
    def __init__(self, params,
                 externally_set_activation_functions=SENTINEL,
                 externally_set_classification_label=SENTINEL):

        super(activation, self).__init__(params)

        assert not (
                (externally_set_activation_functions == SENTINEL and externally_set_classification_label != SENTINEL) or
                (externally_set_activation_functions != SENTINEL and externally_set_classification_label == SENTINEL)),\
            'externally_set_activation_function and classification_label should be provided together'

        self.externally_set_activation_functions = externally_set_activation_functions
        self.externally_set_classification_label = externally_set_classification_label

        self.activation_functions = []
        self.activation_functions_is_set = False

        # print(self.params)

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
                self.activation_functions.append(
                    lambda number_of_active_infected_neighbors, i=node_counter:
                    self.externally_set_activation_function[i][number_of_active_infected_neighbors])
            self.activation_functions_is_set = True
        else:
            assert self.externally_set_classification_label == SENTINEL, \
                'classification_label provided without externally_set_activation_functions'
            self.activation_functions_is_set = False

    def set_activation_probabilities(self):
        """
        This will be called at every time step during the generation of a time series
        to update the activation probabilities based on the number of infected neighbors
        """
        assert self.activation_functions_is_set, 'activation_fucntions are not set'

        current_network = copy.deepcopy(self.params['network'])
        for i in current_network.nodes():
            self.activation_probabilities[self.node_list.index(i)] = \
                self.activation_functions[self.node_list.index(i)](
                    current_network.node[i]['number_of_active_infected_neighbors'])
        self.activation_probabilities_is_set = True

    def step(self):
        self.slow_step()

    # def new_step(self):
    #
    #     assert self.list_of_most_recent_switches_is_updated, \
    #         "list_of_most_recent_switches is mishandled"
    #
    #     self.set_activation_probabilities()
    #
    #     assert self.activation_probabilities_is_set, 'activation_probabilities not set'
    #
    #     current_network = copy.deepcopy(self.params['network'])
    #
    #
    #     list_of_potential_complex_contagion_switches = []
    #
    #     for recent_switch in self.list_of_most_recent_switches:
    #         list_of_potential_complex_contagion_switches = list(set(list_of_potential_complex_contagion_switches)
    #                                                              .union(set(list(current_network.
    #                                                                              neighbors(recent_switch)))))
    #
    #     self.list_of_most_recent_switches = []
    #
    #     if not list_of_potential_complex_contagion_switches: # the spreading process terminates
    #         print('warning the spreading stopped')
    #         self.time_since_infection_is_updated = True
    #         self.time_since_activation_is_updated = True
    #         self.number_of_active_infected_neighbors_is_updated = True
    #
    #     for i in list_of_potential_complex_contagion_switches:
    #         node_i_index = self.node_list.index(i)
    #         self.list_of_most_recent_switches_is_updated = False
    #         # current_network.node[i]['state'] can either be susceptible (0)
    #         # or active infected (0.5) or inactive infected (-0.5)
    #
    #         # transition from susceptible to active infected:
    #
    #         if current_network.node[i]['state'] == susceptible:
    #             assert self.params['network'].node[i]['time_since_infection'] == 0 and \
    #                    self.params['network'].node[i]['time_since_activation'] == 0, 'error: time_since_infection or ' \
    #                                                                                  'time_since_activation mishandled!'
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #
    #             if RD.random() < self.activation_probabilities[node_i_index]:
    #                 self.params['network'].node[i]['state'] = infected*active
    #                 self.list_of_most_recent_switches.append(i)
    #                 for k in self.params['network'].neighbors(i):
    #                     self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
    #             self.number_of_active_infected_neighbors_is_updated = True
    #             self.list_of_most_recent_switches_is_updated = True
    #
    #         # in all the below cases the node is in infected active or infected inactive state.
    #
    #         # transition from active or inactive infected to susceptible:
    #
    #         elif RD.random() < self.params['delta']:
    #             assert 2*abs(current_network.node[i]['state']) == infected, \
    #                 "error: node states are mishandled"
    #             #  here the node should either be active infected (+0.5) or inactive infected (-0.5)
    #             self.params['network'].node[i]['state'] = susceptible
    #
    #             self.list_of_most_recent_switches.append(i)
    #             self.list_of_most_recent_switches_is_updated = True
    #
    #             if current_network.node[i]['state'] == infected*active:
    #                 for k in self.params['network'].neighbors(i):
    #                     assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
    #                         'error: number_of_active_infected_neighbors is mishandled'
    #                     # here number_of_active_infected_neighbors for neighbor k should be at least one
    #                     self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
    #             self.number_of_active_infected_neighbors_is_updated = True
    #             self.params['network'].node[i]['time_since_infection'] = 0
    #             self.params['network'].node[i]['time_since_activation'] = 0
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #
    #         # transition from active infected to inactive infected:
    #
    #         elif current_network.node[i]['state'] == infected*active and RD.random() < self.params['gamma']:
    #             self.params['network'].node[i]['state'] = infected*inactive
    #             self.list_of_most_recent_switches.append(i)
    #             self.list_of_most_recent_switches_is_updated = True
    #             for k in self.params['network'].neighbors(i):
    #                 assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
    #                     'error: number_of_active_infected_neighbors is mishandled'
    #                 # here number_of_active_infected_neighbors for neighbor k should be at least one
    #                 self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
    #             self.number_of_active_infected_neighbors_is_updated = True
    #             self.params['network'].node[i]['time_since_infection'] += 1
    #             self.params['network'].node[i]['time_since_activation'] = 0
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #
    #         # transition from inactive infected to active infected:
    #
    #         elif current_network.node[i]['state'] == infected*inactive and RD.random() < self.params['alpha']:
    #             self.params['network'].node[i]['state'] = infected*active
    #             self.list_of_most_recent_switches.append(i)
    #             self.list_of_most_recent_switches_is_updated = True
    #             for k in self.params['network'].neighbors(i):
    #                 self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
    #             self.number_of_active_infected_neighbors_is_updated = True
    #             self.params['network'].node[i]['time_since_infection'] += 1
    #             self.params['network'].node[i]['time_since_activation'] = 0
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #
    #         # else the node state is either active or inactive infected and
    #         # there are no state transitions, but we still need to update the time_since variables:
    #
    #         else:
    #             self.params['network'].node[i]['time_since_infection'] += 1
    #             if current_network.node[i]['state'] == infected*active:
    #                 self.params['network'].node[i]['time_since_activation'] += 1
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #             self.number_of_active_infected_neighbors_is_updated = True
    #             self.list_of_most_recent_switches_is_updated = True

    def slow_step(self):
        # this step implementation is slow computationally since it performs a comprehensive loop over all nodes

        self.set_activation_probabilities()

        assert self.activation_probabilities_is_set, 'activation_probabilities not set'

        current_network = copy.deepcopy(self.params['network'])
        count_node = 0

        for i in current_network.nodes():
            # current_network.node[i]['state'] can either be susceptible (0)
            # or active infected (0.5) or inactive infected (-0.5)

            # transition from susceptible to active infected:

            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0 and \
                       self.params['network'].node[i]['time_since_activation'] == 0, 'error: time_since_infection or ' \
                                                                                     'time_since_activation mishandled!'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                if RD.random() < self.activation_probabilities[count_node]:
                    self.params['network'].node[i]['state'] = infected * active
                    for k in self.params['network'].neighbors(i):
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True

            # in all the below cases the node is in infected active or infected inactive state.

            # transition from active or inactive infected to susceptible:

            elif RD.random() < self.params['delta']:
                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)
                self.params['network'].node[i]['state'] = susceptible
                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif current_network.node[i]['state'] == infected * active and RD.random() < self.params['gamma']:
                self.params['network'].node[i]['state'] = infected * inactive
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected * inactive and RD.random() < self.params['alpha']:
                self.params['network'].node[i]['state'] = infected * active
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True

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
    def __init__(self, params):
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
                    self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                     ((1/(1 + np.exp((1 / self.params['sigma']) *
                                                            (current_network.node[node_index]['threshold'] -
                                                             number_of_infected_neighbors)))) *
                                                      1.0 * (number_of_infected_neighbors > 0)) +
                                                     0.0 * 1.0 * (number_of_infected_neighbors == 0))
            else:  # not zero at zero
                for i in current_network.nodes():
                    self.activation_functions.append(lambda number_of_infected_neighbors, node_index=i:
                                                     (1/(1 + np.exp((1 / self.params['sigma']) *
                                                            (current_network.node[node_index]['threshold'] -
                                                             number_of_infected_neighbors)))) * 1.0)
        self.activation_functions_is_set = True


class LinearThreshold(activation):
    """
    This is the core threshold model. Probability of adoption below threshold is self.params['fixed_prob'] and
    probability of adoption above threshold is self.params['fixed_prob_high'].
    RandomLinear(), DeterministicLinear(), and RelativeLinear() all inherit this model. Each of these children classes
    have a different way of setting the threshold values for the nodes.
    """
    def __init__(self, params):
        super(LinearThreshold, self).__init__(params)
        self.classification_label = COMPLEX
        self.isLinearThresholdModel = True

    def set_activation_functions(self):
        """
        sets the linear threshold activation functions for each of the nodes
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

    def step(self):

        self.set_activation_probabilities()

        assert self.activation_probabilities_is_set, 'activation_probabilities not set'

        if self.params['fixed_prob'] == 0.0 and self.params['fixed_prob_high'] == 1.0:
            # using the fast update for pure (0,1) complex contagion based on the most recent state switches
            self.fast_update_infected_nodes()
        else:
            self.slow_step()

    def fast_update_infected_nodes(self):
        assert self.list_of_most_recent_switches_is_updated, \
            "list_of_most_recent_switches is mishandled"

        current_network = copy.deepcopy(self.params['network'])

        # current_inactive_infected_nodes = [x for x, y in current_network.nodes(data=True)
        #                                    if y['state'] == infected*inactive]
        #
        # current_active_infected_nodes = [x for x, y in current_network.nodes(data=True)
        #                                  if y['state'] == infected*active]
        # current_susceptible_nodes = [x for x, y in current_network.nodes(data=True)
        #                                  if y['state'] == susceptible]

        list_of_potential_complex_contagion_switches = []

        for recent_switch in self.list_of_most_recent_switches:
            list_of_potential_complex_contagion_switches = list(set(list_of_potential_complex_contagion_switches)
                                                                .union(set(list(current_network.
                                                                                neighbors(recent_switch)))))
        self.list_of_most_recent_switches = []

        if not list_of_potential_complex_contagion_switches:
            # the spreading process terminates
            print('warning the spreading stopped')
            self.time_since_infection_is_updated = True
            self.time_since_activation_is_updated = True
            self.number_of_active_infected_neighbors_is_updated = True

        for i in list_of_potential_complex_contagion_switches:
            node_i_index = self.node_list.index(i)
            self.list_of_most_recent_switches_is_updated = False
            # current_network.node[i]['state'] can either be susceptible (0)
            # or active infected (0.5) or inactive infected (-0.5)

            # transition from susceptible to active infected:

            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0 and \
                       self.params['network'].node[i]['time_since_activation'] == 0, 'error: time_since_infection or ' \
                                                                                     'time_since_activation mishandled!'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                if RD.random() < self.activation_probabilities[node_i_index]:
                    self.params['network'].node[i]['state'] = infected * active
                    self.list_of_most_recent_switches.append(i)
                    for k in self.params['network'].neighbors(i):
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.list_of_most_recent_switches_is_updated = True

            # in all the below cases the node is in infected active or infected inactive state.

            # transition from active or inactive infected to susceptible:

            elif RD.random() < self.params['delta']:
                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)
                self.params['network'].node[i]['state'] = susceptible

                self.list_of_most_recent_switches.append(i)
                self.list_of_most_recent_switches_is_updated = True

                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif current_network.node[i]['state'] == infected * active and RD.random() < self.params['gamma']:
                self.params['network'].node[i]['state'] = infected * inactive
                self.list_of_most_recent_switches.append(i)
                self.list_of_most_recent_switches_is_updated = True
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected * inactive and RD.random() < self.params['alpha']:
                self.params['network'].node[i]['state'] = infected * active
                self.list_of_most_recent_switches.append(i)
                self.list_of_most_recent_switches_is_updated = True
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True
                self.list_of_most_recent_switches_is_updated = True

    # def step(self):
    #
    #     self.set_activation_probabilities()
    #
    #     assert self.activation_probabilities_is_set, 'activation_probabilities not set'
    #
    #     current_network = copy.deepcopy(self.params['network'])
    #
    #     print('we are here')
    #
    #     if self.params['fixed_prob'] == 0.0 and self.params['fixed_prob_high'] == 1.0:
    #     # if the model is pure (0,1) complex contagion then we can loop only over the neighbors of
    #     # list_of_most_recent_switches  - the so-called list_of_potential_complex_contagion_switches
    #     # to speed up computations
    #         print('we are here')
    #
    #         assert self.list_of_most_recent_switches_is_updated, \
    #             "list_of_most_recent_switches is mishandled"
    #
    #
    #         list_of_potential_complex_contagion_switches = []
    #
    #         for recent_switch in self.list_of_most_recent_switches:
    #             list_of_potential_complex_contagion_switches = list(set(list_of_potential_complex_contagion_switches)
    #                                                                  .union(set(list(current_network.
    #                                                                                  neighbors(recent_switch)))))
    #
    #         self.list_of_most_recent_switches = []
    #
    #         if not list_of_potential_complex_contagion_switches: # the spreading process terminates
    #             print('warning the spreading stopped')
    #             self.time_since_infection_is_updated = True
    #             self.time_since_activation_is_updated = True
    #             self.number_of_active_infected_neighbors_is_updated = True
    #
    #         for i in list_of_potential_complex_contagion_switches:
    #             node_i_index = self.node_list.index(i)
    #             self.list_of_most_recent_switches_is_updated = False
    #             # current_network.node[i]['state'] can either be susceptible (0)
    #             # or active infected (0.5) or inactive infected (-0.5)
    #
    #             # transition from susceptible to active infected:
    #
    #             if current_network.node[i]['state'] == susceptible:
    #                 assert self.params['network'].node[i]['time_since_infection'] == 0 and \
    #                        self.params['network'].node[i]['time_since_activation'] == 0, 'error: time_since_infection or ' \
    #                                                                                      'time_since_activation mishandled!'
    #                 self.time_since_infection_is_updated = True
    #                 self.time_since_activation_is_updated = True
    #
    #                 if RD.random() < self.activation_probabilities[node_i_index]:
    #                     self.params['network'].node[i]['state'] = infected*active
    #                     self.list_of_most_recent_switches.append(i)
    #                     for k in self.params['network'].neighbors(i):
    #                         self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
    #                 self.number_of_active_infected_neighbors_is_updated = True
    #                 self.list_of_most_recent_switches_is_updated = True
    #
    #             # in all the below cases the node is in infected active or infected inactive state.
    #
    #             # transition from active or inactive infected to susceptible:
    #
    #             elif RD.random() < self.params['delta']:
    #                 assert 2*abs(current_network.node[i]['state']) == infected, \
    #                     "error: node states are mishandled"
    #                 #  here the node should either be active infected (+0.5) or inactive infected (-0.5)
    #                 self.params['network'].node[i]['state'] = susceptible
    #
    #                 self.list_of_most_recent_switches.append(i)
    #                 self.list_of_most_recent_switches_is_updated = True
    #
    #                 if current_network.node[i]['state'] == infected*active:
    #                     for k in self.params['network'].neighbors(i):
    #                         assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
    #                             'error: number_of_active_infected_neighbors is mishandled'
    #                         # here number_of_active_infected_neighbors for neighbor k should be at least one
    #                         self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
    #                 self.number_of_active_infected_neighbors_is_updated = True
    #                 self.params['network'].node[i]['time_since_infection'] = 0
    #                 self.params['network'].node[i]['time_since_activation'] = 0
    #                 self.time_since_infection_is_updated = True
    #                 self.time_since_activation_is_updated = True
    #
    #             # transition from active infected to inactive infected:
    #
    #             elif current_network.node[i]['state'] == infected*active and RD.random() < self.params['gamma']:
    #                 self.params['network'].node[i]['state'] = infected*inactive
    #                 self.list_of_most_recent_switches.append(i)
    #                 self.list_of_most_recent_switches_is_updated = True
    #                 for k in self.params['network'].neighbors(i):
    #                     assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
    #                         'error: number_of_active_infected_neighbors is mishandled'
    #                     # here number_of_active_infected_neighbors for neighbor k should be at least one
    #                     self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
    #                 self.number_of_active_infected_neighbors_is_updated = True
    #                 self.params['network'].node[i]['time_since_infection'] += 1
    #                 self.params['network'].node[i]['time_since_activation'] = 0
    #                 self.time_since_infection_is_updated = True
    #                 self.time_since_activation_is_updated = True
    #
    #             # transition from inactive infected to active infected:
    #
    #             elif current_network.node[i]['state'] == infected*inactive and RD.random() < self.params['alpha']:
    #                 self.params['network'].node[i]['state'] = infected*active
    #                 self.list_of_most_recent_switches.append(i)
    #                 self.list_of_most_recent_switches_is_updated = True
    #                 for k in self.params['network'].neighbors(i):
    #                     self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
    #                 self.number_of_active_infected_neighbors_is_updated = True
    #                 self.params['network'].node[i]['time_since_infection'] += 1
    #                 self.params['network'].node[i]['time_since_activation'] = 0
    #                 self.time_since_infection_is_updated = True
    #                 self.time_since_activation_is_updated = True
    #
    #             # else the node state is either active or inactive infected and
    #             # there are no state transitions, but we still need to update the time_since variables:
    #
    #             else:
    #                 self.params['network'].node[i]['time_since_infection'] += 1
    #                 if current_network.node[i]['state'] == infected*active:
    #                     self.params['network'].node[i]['time_since_activation'] += 1
    #                 self.time_since_infection_is_updated = True
    #                 self.time_since_activation_is_updated = True
    #                 self.number_of_active_infected_neighbors_is_updated = True
    #                 self.list_of_most_recent_switches_is_updated = True


class RandomLinear(LinearThreshold):
    """
    Implements a "random" linear threshold activation function. Thresholds are set relative to the total number of
    neighbors (ratios of infected in the local neighborhoods). The threshold values are set uniformly at random from the
    (0,1) interval. Above the threshold infection happens with fixed probability 'fixed_prob_high'. Below the threshold
    infection happens with fixed probability 'fixed_prob'. The fixed probabilities can be set to one or zero.
    """
    def __init__(self,params):
        super(RandomLinear, self).__init__(params)
        self.classification_label = COMPLEX
        # self.isLinearThresholdModel = True

        assert 'thresholds' not in self.fixed_params, \
            "thresholds should not be supplied for RandomLinear contagion model"

        # set the thresholds for the RandomLinear model:

        relative_thresholds = list(np.random.uniform(size=self.params['size']))
        all_degrees = self.params['network'].degree()
        self.params['thresholds'] = list(map(lambda x, y: float(x[1]) * y, all_degrees, relative_thresholds))


class RelativeLinear(LinearThreshold):
    """
    Implements a linear threshold activation function. Thresholds are set relative to the total number of neighbors
    (ratios of infected in the local neighborhoods). The threshold values are set fixed. Above the threshold infection happens with fixed probability 'fixed_prob_high'. Below the threshold
    infection happens with fixed probability 'fixed_prob'. The fixed probabilities can be set to one or zero.
    """
    def __init__(self, params):
        super(RelativeLinear, self).__init__(params)
        self.classification_label = COMPLEX
        # self.isRelativeThresholdModel = True

        # setting the absolute thresholds based on the specified relative threshold value

        assert 'relative_threshold' in self.fixed_params, \
            "Relative threshold should be supplied for RelativeLinear contagion model."
        relative_thresholds = [self.params['relative_threshold']] * self.params['size']
        all_degrees = self.params['network'].degree()
        self.params['thresholds'] = list(map(lambda x, y: float(x[1]) * y, all_degrees, relative_thresholds))


class DeterministicLinear(LinearThreshold):
    """
    Similar to the  linear threshold model except that thresholds are not ratios and are not random. 'thresholds' are
    fixed and they can be set all the same equal to theta.
    """
    def __init__(self, params):
        super(DeterministicLinear, self).__init__(params)
        self.classification_label = COMPLEX

        # setting the fixed absolute threshold for the DeterministicLinear Model

        assert 'theta' in self.fixed_params, \
            "Theta should be supplied for DeterministicLinear contagion model."
        self.params['thresholds'] = [self.params['theta']] * self.params['size']


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
                assert self.params['network'].node[i]['time_since_infection'] == 0 \
                       and self.params['network'].node[i]['time_since_activation'] == 0, 'error: ' \
                                                                                         'time_since_infection or ' \
                                                                                         'timr_since_activation ' \
                                                                                         'mishandle'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                if (current_network.node[i]['threshold'] <=
                        current_network.node[i]['number_of_active_infected_neighbors']):
                    # print('we are here')
                    if RD.random() < self.params['fixed_prob_high']:
                        self.params['network'].node[i]['state'] = infected*active
                        for k in self.params['network'].neighbors(i):
                            self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1

                else:  # if the node cannot be infected through complex contagion
                    # see if it can be infected through simple contagion along cycle edges
                    for j in current_network.neighbors(i):

                        j_is_a_cycle_neighbor = ((abs(i - j) == 1) or
                                                 (abs(i - j) == (self.params['size'] - 1)))

                        if (current_network.node[j]['state'] == infected*active
                                and j_is_a_cycle_neighbor):
                            if RD.random() < self.params['fixed_prob']:
                                self.params['network'].node[i]['state'] = infected*active
                                for k in self.params['network'].neighbors(i):
                                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                                break

                self.number_of_active_infected_neighbors_is_updated = True
                # the updating should happen in
                # self.params['network'] not the current network

            # if node i is already infected (infected active or infected inactive) but recovers
            elif RD.random() < self.params['delta']:

                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)

                self.params['network'].node[i]['state'] = susceptible

                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif current_network.node[i]['state'] == infected*active and RD.random() < self.params['gamma']:
                self.params['network'].node[i]['state'] = infected*inactive
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected*inactive and RD.random() < self.params['alpha']:
                self.params['network'].node[i]['state'] = infected * active
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True


class SimpleOnlyAlongOriginalEdges(contagion_model):
    """
    Implements an a special contagion model that is useful for measuring spread time over real networks when studying
    the effect of edge addition interventions. It allows simple contagion along only the edges of the original network.
    The complex contagion happens along all edges (original and added) as before.
    """

    def __init__(self, params):
        super(SimpleOnlyAlongOriginalEdges, self).__init__(params)
        self.classification_label = COMPLEX
        assert self.params['add_edges'] or self.params['rewire'], \
            "This contagion model is only suitable when there is some edge addition or " \
            "rewiring intervention done to the original network."
        assert self.params['original_network'] is not None, \
            "original_network should be supplied to work with SimpleOnlyAlongOriginalEdges(contagion_model)"

    def step(self):
        current_network = copy.deepcopy(self.params['network'])
        for i in current_network.nodes():

            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0 \
                       and self.params['network'].node[i]['time_since_activation'] == 0, 'error: ' \
                                                                                         'time_since_infection or ' \
                                                                                         'timr_since_activation ' \
                                                                                         'mishandle'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                if (current_network.node[i]['threshold'] <=
                        current_network.node[i]['number_of_active_infected_neighbors']):
                    # print('we are here')
                    if RD.random() < self.params['fixed_prob_high']:
                        self.params['network'].node[i]['state'] = infected*active
                        for k in self.params['network'].neighbors(i):
                            self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1

                else:  # if the node cannot be infected through complex contagion
                    # see if it can be infected through simple contagion along the "original" edges
                    for j in current_network.neighbors(i):

                        j_is_an_original_neighbor = self.params['original_network'].has_edge(i, j)

                        if (current_network.node[j]['state'] == infected*active
                                and j_is_an_original_neighbor):
                            if RD.random() < self.params['fixed_prob']:
                                self.params['network'].node[i]['state'] = infected*active
                                for k in self.params['network'].neighbors(i):
                                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                                break

                self.number_of_active_infected_neighbors_is_updated = True
                # the updating should happen in
                # self.params['network'] not the current network

            # if node i is already infected (infected active or infected inactive) but recovers
            elif RD.random() < self.params['delta']:

                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)

                self.params['network'].node[i]['state'] = susceptible

                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif current_network.node[i]['state'] == infected*active and RD.random() < self.params['gamma']:
                self.params['network'].node[i]['state'] = infected*inactive
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected*inactive and RD.random() < self.params['alpha']:
                self.params['network'].node[i]['state'] = infected * active
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True


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

            # current_network.node[i]['state'] can either be susceptible (0)
            # or active infected (0.5) or inactive infected (-0.5)

            # transition from susceptible to active infected:

            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0 and \
                       self.params['network'].node[i][
                           'time_since_activation'] == 0, 'error: time_since_infection or ' \
                                                          'time_since_activation mishandled!'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                for j in current_network.neighbors(i):
                    if current_network.node[j]['state'] == infected*active:
                        assert current_network.node[j]['time_since_activation'] < self.memory, \
                            "error: should not remain activation beyond mem times."
                        if RD.random() < self.params['beta']:
                            self.params['network'].node[i]['state'] = infected*active
                            for k in self.params['network'].neighbors(i):
                                self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                            break
                self.number_of_active_infected_neighbors_is_updated = True

            # in all the below cases the node is in infected active or infected inactive state.

            # transition from active or inactive infected to susceptible:

            elif RD.random() < self.params['delta']:
                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)

                assert self.params['network'].node[i]['time_since_activation'] <= self.memory, \
                    "error: time_since_activation should not get greater than mem"
                self.params['network'].node[i]['state'] = susceptible
                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif (current_network.node[i]['state'] == infected*active
                  and (RD.random() < self.params['gamma'] or
                       current_network.node[i]['time_since_activation'] == self.memory)):
                self.params['network'].node[i]['state'] = infected*inactive
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected * inactive and RD.random() < self.params['alpha']:
                assert self.params['network'].node[i]['time_since_activation'] == 0, \
                    "error: time_since_activation should be zero for an inactive node"
                self.params['network'].node[i]['state'] = infected * active
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                assert self.params['network'].node[i]['time_since_activation'] < self.memory, \
                    "error: time_since_activation should be less than mem"
                if current_network.node[i]['state'] == infected * inactive:
                    assert self.params['network'].node[i]['time_since_activation'] == 0,\
                        "error: time_since_activation should be zero for an inactive node"
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True


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
