# Comparing the rate of contagion in the original and rewired network.
# loops over all files in
# uses maslov_sneppen_rewiring(num_steps = np.floor(0.1 * self.params['network'].number_of_edges()))
# to rewire the network
# uses avg_speed_of_spread(dataset_size=1000,cap=0.9, mode='max') to measure the rate of spread

from settings import *

# assert settings.data_dump, "we should be in data_dump mode!"

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

# network_id_list = list(np.linspace(1, 17, TOP_ID))  # cannot do 152
#
# network_id_list = [str(int(id)) for id in network_id_list]

rewiring_percentage_list = [0, 5, 10, 15, 20, 25]
# loop_mode = (len(rewiring_percentage_list) > 1)
# print(loop_mode)

percent_more_edges_list = [5, 10, 15, 20, 25]


def check_type(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


if __name__ == '__main__':

    for network_id in network_id_list:

        print('network id', network_id)

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
            print('warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
            print('number of edges before self loop removal: ', G.size())
            G.remove_edges_from(G.selfloop_edges())
            print('number of edges before self loop removal: ', G.size())

        network_size = NX.number_of_nodes(G)

        if do_computations:

            # spreading time computations for rewiring interventions

            for rewiring_percentage in rewiring_percentage_list:
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
                        'num_edges_for_random_random_rewiring': 0.01*rewiring_percentage * G.number_of_edges(),
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

            # spreading time computations for edge addition interventions

            for percent_more_edges in percent_more_edges_list:
                print('network id', network_id, 'edge addition: ', percent_more_edges)
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
                    pickle.dump(speed_samples_add_random,
                                open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                     + str(percent_more_edges) + '_percent_' + 'add_random_'
                                     + network_group + network_id
                                     + model_id + '.pkl', 'wb'))
                    pickle.dump(speed_samples_add_triad, open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                              + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                              + network_group + network_id
                                                              + model_id + '.pkl', 'wb'))