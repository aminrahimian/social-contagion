# compute the spread time in the original network and under edge addition and rewiring interventions


from settings import *

assert do_computations, "we should be in do_computations mode"

from models import *



from multiprocessing import Pool
#
# class Engine(object):
#     def __init__(self, parameters):
#         self.parameters = parameters
#     def __call__(self, filename):
#         sci = fits.open(filename + '.fits')
#         manipulated = manipulate_image(sci, self.parameters)
#         return manipulated
#
# try:
#     pool = Pool(8) # on 8 processors
#     engine = Engine(my_parameters)
#     data_outputs = pool.map(engine, data_inputs)
# finally: # To make sure processes are closed in the end, even if errors happen
#     pool.close()
#     pool.join()

size_of_dataset = 2

rewiring_percentage_list = [0, 5, 10, 15, 20, 25]

percent_more_edges_list = [5, 10, 15, 20, 25]

intervention_type = 'triad-addition' #, 'random-addition', 'rewiring' #

number_initial_seeds = 2

def measure_rewiring_spread_time(network_id, rewiring_percentage):
    #  load in the network and extract preliminary data
    fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')
    G = NX.read_edgelist(fh, delimiter=DELIMITER)
    #  get the largest connected component:
    if not NX.is_connected(G):
        G = max(NX.connected_component_subgraphs(G), key=len)
        print('largest connected component extracted with size ', len(G.nodes()))
    #  remove self loops:
    if len(list(G.selfloop_edges())) > 0:
        print(
            'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
        print('number of edges before self loop removal: ', G.size())
        G.remove_edges_from(G.selfloop_edges())
        print('number of edges before self loop removal: ', G.size())
    network_size = NX.number_of_nodes(G)

    if rewiring_percentage is 0:
        print('network id', network_id, 'original')
        params_original = {
            'network': G,
            'size': network_size,
            'add_edges': False,
            'initialization_mode': 'fixed_number_initial_infection',
            'initial_infection_number': number_initial_seeds,
            'delta': 0.0000000000000001,
            'fixed_prob_high': fixed_prob_high,
            'fixed_prob': fixed_prob_low,
            'alpha': alpha,
            'gamma': gamma,
            'theta': 2,
            'rewire': False,
            'rewiring_mode': 'random_random',
            'num_edges_for_random_random_rewiring': None,
        }

        dynamics_original = DeterministicLinear(params_original)
        speed_original, std_original, _, _, speed_samples_original = \
            dynamics_original.avg_speed_of_spread(
                dataset_size=size_of_dataset,
                cap=0.9,
                mode='max')
        print(speed_original, std_original)
        print(speed_samples_original)
        print(type(speed_original))
        if save_computations:
            pickle.dump(speed_samples_original, open(spreading_pickled_samples_directory_address
                                                     + 'speed_samples_original_'
                                                     + network_group + network_id
                                                     + model_id + '.pkl', 'wb'))
    else:  # rewiring

        print('network id', network_id, 'rewiring: ', rewiring_percentage)

        params_rewired = {
            'network': G,
            'size': network_size,
            'add_edges': False,
            'initialization_mode': 'fixed_number_initial_infection',
            'initial_infection_number': number_initial_seeds,
            'delta': 0.0000000000000001,
            'fixed_prob_high': fixed_prob_high,
            'fixed_prob': fixed_prob_low,
            'alpha': alpha,
            'gamma': gamma,
            'theta': 2,
            'rewire': True,
            'rewiring_mode': 'random_random',
            'num_edges_for_random_random_rewiring': 0.01 * rewiring_percentage * G.number_of_edges(),
        }

        dynamics_rewired = DeterministicLinear(params_rewired)
        speed_rewired, std_rewired, _, _, speed_samples_rewired = \
            dynamics_rewired.avg_speed_of_spread(dataset_size=size_of_dataset,
                                                 cap=0.9,
                                                 mode='max')
        print(speed_rewired, std_rewired)
        print(speed_samples_rewired)
        print(NX.is_connected(G))
        if save_computations:
            pickle.dump(speed_samples_rewired,
                        open(spreading_pickled_samples_directory_address + 'speed_samples_'
                             + str(rewiring_percentage) +
                             '_percent_rewiring_' + network_group + network_id
                             + model_id + '.pkl', 'wb'))
    return


def measure_triad_addition_spread_time(network_id, percent_more_edges):
    print('network id', network_id, 'traid edge addition: ', percent_more_edges)
    #  load in the network and extract preliminary data
    fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')
    G = NX.read_edgelist(fh, delimiter=DELIMITER)
    print('original size ', len(G.nodes()))
    #  get the largest connected component:
    if not NX.is_connected(G):
        G = max(NX.connected_component_subgraphs(G), key=len)
        print('largest connected component extracted with size ', len(G.nodes()))
    #  remove self loops:
    if len(list(G.selfloop_edges())) > 0:
        print(
            'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
        print('number of edges before self loop removal: ', G.size())
        G.remove_edges_from(G.selfloop_edges())
        print('number of edges before self loop removal: ', G.size())
    network_size = NX.number_of_nodes(G)

    params_add_triad = {
        'network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'triadic_closures',
        'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
        'initialization_mode': 'fixed_number_initial_infection',
        'initial_infection_number': number_initial_seeds,
        'delta': 0.0000000000000001,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        'alpha': alpha,
        'gamma': gamma,
        'theta': 2,
        'rewire': False,
    }

    dynamics_add_triad = DeterministicLinear(params_add_triad)

    speed_add_triad, std_add_triad, _, _, speed_samples_add_triad = \
        dynamics_add_triad.avg_speed_of_spread(
            dataset_size=size_of_dataset,
            cap=0.9,
            mode='max')
    print(speed_add_triad, std_add_triad)
    print(speed_samples_add_triad)
    print(NX.is_connected(G))
    if save_computations:
        pickle.dump(speed_samples_add_triad, open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                  + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                  + network_group + network_id
                                                  + model_id + '.pkl', 'wb'))
    return


def measure_random_addition_spread_time(network_id,percent_more_edges):
    print('network id', network_id, 'traid edge addition: ', percent_more_edges)
    #  load in the network and extract preliminary data
    fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')
    G = NX.read_edgelist(fh, delimiter=DELIMITER)
    print('original size ', len(G.nodes()))
    #  get the largest connected component:
    if not NX.is_connected(G):
        G = max(NX.connected_component_subgraphs(G), key=len)
        print('largest connected component extracted with size ', len(G.nodes()))
    #  remove self loops:
    if len(list(G.selfloop_edges())) > 0:
        print(
            'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
        print('number of edges before self loop removal: ', G.size())
        G.remove_edges_from(G.selfloop_edges())
        print('number of edges before self loop removal: ', G.size())
    network_size = NX.number_of_nodes(G)

    params_add_random = {
        'network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'random',
        'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
        'initialization_mode': 'fixed_number_initial_infection',
        'initial_infection_number': number_initial_seeds,
        'delta': 0.0000000000000001,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        'alpha': alpha,
        'gamma': gamma,
        'theta': 2,
        'rewire': False,
    }

    dynamics_add_random = DeterministicLinear(params_add_random)

    speed_add_random, std_add_random, _, _, speed_samples_add_random = \
        dynamics_add_random.avg_speed_of_spread(
            dataset_size=size_of_dataset,
            cap=0.9,
            mode='max')
    print(speed_add_random, std_add_random)
    print(speed_samples_add_random)
    if save_computations:
        pickle.dump(speed_samples_add_random,
                    open(spreading_pickled_samples_directory_address + 'speed_samples_'
                         + str(percent_more_edges) + '_percent_' + 'add_random_'
                         + network_group + network_id
                         + model_id + '.pkl', 'wb'))
    return

if __name__ == '__main__':

    if intervention_type == 'rewiring':
        # spreading time computations for rewiring interventions
        for network_id in network_id_list:
            for rewiring_percentage in rewiring_percentage_list:
                measure_rewiring_spread_time(network_id,rewiring_percentage)
    elif intervention_type == 'triad-addition':
        # spreading time computations for edge addition interventions
        for network_id in network_id_list:
            for percent_more_edges in percent_more_edges_list:
                measure_triad_addition_spread_time(network_id, percent_more_edges)
    elif intervention_type == 'random-addition':
        # spreading time computations for edge addition interventions
        for network_id in network_id_list:
            for percent_more_edges in percent_more_edges_list:
                measure_random_addition_spread_time(network_id, percent_more_edges)