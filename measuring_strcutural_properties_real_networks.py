# Computes properties of real networks both original networks and
# its modifications under various structural interventions


from models import *
from multiprocessing import Pool

size_of_dataset = 500

intervention_size_list = [10] #[5, 10, 15, 20, 25]

old_properties = ['avg_clustering','average_shortest_path_length', 'diameter', 'size_2_core']
# when only need to calculate the fb100 avg_clustering:
# old_properties = ['avg_clustering']

new_properties = ['avg_degree','diam_2_core', 'max_degree', 'min_degree',
                  'max_degree_2_core', 'min_degree_2_core',
                  'avg_degree_2_core', 'number_edges', 'number_edges_2_core',
                  'avg_clustering_2_core', 'transitivity', 'transitivity_2_core',
                  'num_leaves']

all_properties = old_properties + new_properties
# when only need to calculate the fb100 avg_clustering:
# all_properties = old_properties

included_properties = all_properties

theta = random.choice([2, 3, 4, 5]) #none of the properties depend investigated depend on the value of theta
# when calculate the fb100 avg_clustering: theta = [0.25, 0.25, 0.25, 0.25]

generate_network_intervention_dataset = True
# determines to whether generate networks (true) or
# load and use previously generated networks (False)


def check_type_is_str(obj):
    if not isinstance(obj, str):
        return str(obj)
    else:
        return obj


def generate_network_intervention_datasets(network_id, intervention_size):
    print('intervention size:', intervention_size, 'network_id', network_id)

    #  load in the network and extract preliminary data

    fh = open(edgelist_directory_address + network_group + check_type_is_str(network_id) + '.txt', 'rb')

    G = NX.read_edgelist(fh, delimiter=DELIMITER)

    print('original size ', len(G.nodes()))

    if not NX.is_connected(G):
        G = max(NX.connected_component_subgraphs(G), key=len)
        print('largest connected component extracted with size ', len(G.nodes()))

    if len(list(G.selfloop_edges())) > 0:
        print(
            'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed.')
        print('number of edges before self loop removal: ', G.size())
        G.remove_edges_from(G.selfloop_edges())
        print('number of edges before self loop removal: ', G.size())

    network_size = NX.number_of_nodes(G)

    params_add_random = {
        'network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'random',
        'number_of_edges_to_be_added': int(np.floor(0.01 * intervention_size * G.number_of_edges())),
        # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
        'initialization_mode': 'fixed_number_initial_infection',
        # 'initial_infection_number': number_initial_seeds,
        'delta': delta,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        # 'theta': theta,
        'theta_distribution': theta,
        'rewire': False,
    }

    dynamics_add_random = ProbabilityDistributionLinear(params_add_random)
    # dynamics_add_random = DeterministicLinear(params_add_random)

    params_add_triad = {
        'network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'triadic_closures',
        'number_of_edges_to_be_added': int(np.floor(0.01 * intervention_size * G.number_of_edges())),
        # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
        'initialization_mode': 'fixed_number_initial_infection',
        # 'initial_infection_number': number_initial_seeds,
        'delta': delta,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        # 'theta': theta,
        'theta_distribution': theta,
        'rewire': False,
        # rewire 10% of edges
    }

    # dynamics_add_triad = DeterministicLinear(params_add_triad)
    dynamics_add_triad = ProbabilityDistributionLinear(params_add_triad)

    params_rewired = {
        'network': G,
        'size': network_size,
        'add_edges': False,
        # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
        'initialization_mode': 'fixed_number_initial_infection',
        # 'initial_infection_number': number_initial_seeds,
        'delta': delta,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        # 'theta': theta,
        'theta_distribution': theta,
        'rewire': True,
        'rewiring_mode': 'random_random',
        'num_edges_for_random_random_rewiring': 0.01 * intervention_size * G.number_of_edges(),
        # rewire 15% of edges
    }

    dynamics_rewired = ProbabilityDistributionLinear(params_rewired)
    # dynamics_rewired = DeterministicLinear(params_rewired)

    add_random_networks = \
        dynamics_add_random.generate_network_intervention_dataset(dataset_size=size_of_dataset)

    print('generated add_random_networks')

    add_triad_networks = \
        dynamics_add_triad.generate_network_intervention_dataset(dataset_size=size_of_dataset)

    print('generated add_triad_networks')

    rewired_networks = \
        dynamics_rewired.generate_network_intervention_dataset(dataset_size=size_of_dataset)

    print('generated rewired_networks.')

    if save_computations:
        pickle.dump(add_random_networks,
                    open(networks_pickled_samples_directory_address + 'networks_'
                         + str(intervention_size) + '_percent_' + 'add_random_'
                         + network_group + network_id + '.pkl', 'wb'))

        pickle.dump(add_triad_networks,
                    open(networks_pickled_samples_directory_address + 'networks_'
                         + str(intervention_size) + '_percent_' + 'add_triad_'
                         + network_group + network_id + '.pkl', 'wb'))

        pickle.dump(rewired_networks,
                    open(networks_pickled_samples_directory_address + 'networks_'
                         + str(intervention_size) + '_percent_' + 'rewiring_'
                         + network_group + network_id + '.pkl', 'wb'))

        print('picked networks in ' + networks_pickled_samples_directory_address)


def measure_properties_of_network_intervention_datasets(network_id, intervention_size):
    print('intervention size:', intervention_size, 'network id:', network_id)

    add_random_networks = pickle.load(open(networks_pickled_samples_directory_address + 'networks_'
                                           + str(intervention_size) + '_percent_' + 'add_random_'
                                           + network_group + network_id + '.pkl', 'rb'))
    print('loaded add_random_networks from ' + networks_pickled_samples_directory_address)

    add_triad_networks = pickle.load(open(networks_pickled_samples_directory_address + 'networks_'
                                          + str(intervention_size) + '_percent_' + 'add_triad_'
                                          + network_group + network_id + '.pkl', 'rb'))

    print('loaded add_triad_networks from ' + networks_pickled_samples_directory_address)

    rewired_networks = pickle.load(open(networks_pickled_samples_directory_address + 'networks_'
                                        + str(intervention_size) + '_percent_' + 'rewiring_'
                                        + network_group + network_id + '.pkl', 'rb'))

    print('loaded rewired_networks from ' + networks_pickled_samples_directory_address)

    # compute properties on the loaded datasets

    properties_sample_add_random = []
    properties_sample_add_triad = []
    properties_sample_rewired = []

    for included_property in included_properties:
        properties_sample_add_random += [measure_property(add_random_networks,
                                                          included_property, size_of_dataset)]
        properties_sample_add_triad += [measure_property(add_triad_networks,
                                                         included_property, size_of_dataset)]
        properties_sample_rewired += [measure_property(rewired_networks,
                                                       included_property, size_of_dataset)]
    if save_computations:
        for included_property in included_properties:
            print(included_property, properties_sample_add_random[included_properties.index(included_property)])
            pickle.dump(properties_sample_add_random[included_properties.index(included_property)],
                        open(properties_pickled_samples_directory_address + included_property + '_'
                             + str(intervention_size) + '_percent_' + 'add_random_'
                             + network_group + network_id + '.pkl', 'wb'))
            print(included_property,
                  properties_sample_add_triad[included_properties.index(included_property)])
            pickle.dump(properties_sample_add_triad[included_properties.index(included_property)],
                        open(properties_pickled_samples_directory_address + included_property + '_'
                             + str(intervention_size) + '_percent_' + 'add_triad_'
                             + network_group + network_id + '.pkl', 'wb'))
            print(included_property,
                  properties_sample_rewired[included_properties.index(included_property)])
            pickle.dump(properties_sample_rewired[included_properties.index(included_property)],
                        open(properties_pickled_samples_directory_address + included_property + '_'
                             + str(intervention_size) + '_percent_' + 'rewiring_'
                             + network_group + network_id + '.pkl', 'wb'))
            print('pickled ' + included_property + '_'
                  + str(intervention_size) + '_percent_' + 'interventions_'
                  + network_group + network_id)

        print('pickled all properties.')


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode!"

    # generate intervention network structures:
    if generate_network_intervention_dataset:
        if do_multiprocessing:
            with multiprocessing.Pool(processes = 4) as pool:
            # with multiprocessing.Pool(processes=number_CPU) as pool:
                pool.starmap(generate_network_intervention_datasets, product(network_id_list, intervention_size_list))
        else:  # not multiprocessing, do for-loops
            for intervention_size in intervention_size_list:
                for network_id in network_id_list:
                    generate_network_intervention_datasets(network_id, intervention_size)

    # compute the properties on the network intervention datasets:

    if do_multiprocessing:
        # with multiprocessing.Pool(processes=number_CPU) as pool:
        with multiprocessing.Pool(processes=4) as pool:
            pool.starmap(measure_properties_of_network_intervention_datasets,
                         product(network_id_list, intervention_size_list))
            pool.close()
            pool.join()
    else:  # not multiprocessing, do for-loops
        for intervention_size in intervention_size_list:
            for network_id in network_id_list:
                measure_properties_of_network_intervention_datasets(network_id, intervention_size)

    # pool.close()
    # pool.join()
