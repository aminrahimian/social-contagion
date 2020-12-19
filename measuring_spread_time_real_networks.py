# compute the spread time in the original network and under edge addition and rewiring interventions

from models import *
import multiprocessing
from multiprocessing import Pool
from pathlib import Path

VERBOSE = True

CHECK_FOR_EXISTING_PKL_SAMPLES = False

size_of_dataset = 10

rewiring_percentage_list = [10]

percent_more_edges_list = [10]

do_computations_for_original_network = True

intervention_type = 'all'
#'triad-addition' #'all'  # 'rewiring' #'all'  # 'triad-addition'  # 'random-addition'# 'rewiring' #

all_interventions = ['triad-addition', 'random-addition', 'rewiring']

addition_interventions = ['triad-addition', 'random-addition']

number_initial_seeds = 2

CAP = 0.9

def measure_rewiring_spread_time(network_id, rewiring_percentage, theta):
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

        if CHECK_FOR_EXISTING_PKL_SAMPLES:
            path = Path(spreading_pickled_samples_directory_address + 'infection_size_original_'
                        + network_group + network_id
                        + model_id + '_' + str(theta) + '.pkl')
            if path.is_file():
                print('infection_size_original_'
                      + network_group + network_id
                      + model_id + '_' + str(theta) + ' already exists')
                return

        params_original = {
            'network': G,
            'original_network': G,
            'size': network_size,
            'add_edges': False,
            'initialization_mode': 'fixed_number_initial_infection',
            'initial_infection_number': number_initial_seeds,
            'delta': delta,
            'fixed_prob_high': fixed_prob_high,
            'fixed_prob': fixed_prob_low,
            'alpha': alpha,
            'gamma': gamma,
            # 'theta': theta,
            'theta_distribution': theta,
            'rewire': False,
            'rewiring_mode': 'random_random',
            'num_edges_for_random_random_rewiring': None,
        }
        if model_id == '_model_4':
            dynamics_original = SimpleOnlyAlongOriginalEdges(params_original)
        elif model_id == '_model_5':
            params_original['relative_threshold'] = zeta
            dynamics_original = RelativeLinear(params_original)
        elif model_id in ['_model_1', '_model_2', '_model_3', '_model_6', '_model_7']:
            # dynamics_original = DeterministicLinear(params_original)
            dynamics_original = ProbabilityDistributionLinear(params_original)
        else:
            print('model_id is not valid')
            exit()
        speed_original, std_original, _, _, speed_samples_original, \
            infection_size_original, infection_size_std_original, _, _, infection_size_samples_original, fractional_evolution_original = \
            dynamics_original.avg_speed_of_spread(
                dataset_size=size_of_dataset,
                cap=CAP,
                mode='max')
        if VERBOSE:
            print('mean time to spread:', speed_original, std_original)
            print('mean infection_size:', infection_size_original, infection_size_std_original)
            print('spread time samples:', speed_samples_original)
            print('infection size samples:', infection_size_samples_original)
            print('Fractional serie:',fractional_evolution_original)

        if save_computations:

            pickle.dump(speed_samples_original, open(spreading_pickled_samples_directory_address
                                                     + 'speed_samples_original_'
                                                     + network_group + network_id
                                                     + model_id + '_' + str(theta) + '.pkl', 'wb'))

            pickle.dump(infection_size_samples_original, open(spreading_pickled_samples_directory_address
                                                              + 'infection_size_original_'
                                                              + network_group + network_id
                                                              + model_id + '_' + str(theta) + '.pkl', 'wb'))

            pickle.dump(fractional_evolution_original, open(spreading_pickled_samples_directory_address
                                                              + 'fractional_evolution_original_'
                                                              + network_group + network_id
                                                              + model_id + '_' + str(theta) + '.pkl', 'wb'))

    else:  # rewiring

        print('network id', network_id, 'rewiring: ', rewiring_percentage)

        if CHECK_FOR_EXISTING_PKL_SAMPLES:
            path = Path(spreading_pickled_samples_directory_address + 'infection_size_samples_'
                        + str(rewiring_percentage) + '_percent_rewiring_'
                        + network_group + network_id
                        + model_id + '_' + str(theta) + '.pkl')
            if path.is_file():
                print('infection_size_samples_' + str(rewiring_percentage) + '_percent_rewiring_'
                      + network_group + network_id
                      + model_id + '_' + str(theta) + ' already exists')
                return

        params_rewired = {
            'network': G,
            'original_network': G,
            'size': network_size,
            'add_edges': False,
            'initialization_mode': 'fixed_number_initial_infection',
            'initial_infection_number': number_initial_seeds,
            'delta': delta,
            'fixed_prob_high': fixed_prob_high,
            'fixed_prob': fixed_prob_low,
            'alpha': alpha,
            'gamma': gamma,
            # 'theta': theta,
            'theta_distribution': theta,
            'rewire': True,
            'rewiring_mode': 'random_random',
            'num_edges_for_random_random_rewiring': 0.01 * rewiring_percentage * G.number_of_edges(),
        }

        if model_id == '_model_4':
            dynamics_rewired = SimpleOnlyAlongOriginalEdges(params_rewired)
        elif model_id == '_model_5':
            params_rewired['relative_threshold'] = zeta
            dynamics_rewired = RelativeLinear(params_rewired)
        elif model_id in ['_model_1', '_model_2', '_model_3', '_model_6', '_model_7']:
            # dynamics_rewired = DeterministicLinear(params_rewired)
            dynamics_rewired = ProbabilityDistributionLinear(params_rewired)
        else:
            print('model_id is not valid')
            exit()

        speed_rewired, std_rewired, _, _, speed_samples_rewired, \
            infection_size_rewired, infection_size_std_rewired, _, _, infection_size_samples_rewired, fraction_evolution_rewired = \
            dynamics_rewired.avg_speed_of_spread(dataset_size=size_of_dataset,
                                                 cap=CAP,
                                                 mode='max')

        if VERBOSE:
            print('mean spreading time in the rewired network:', speed_rewired, std_rewired)
            print('spreading time samples in the rewired network:', speed_samples_rewired)
            print('mean infection size in the rewired network:', infection_size_rewired, std_rewired)
            print('infection size samples in the rewired network:', infection_size_samples_rewired)
            print('fractional evolution in the rewired network:', fraction_evolution_rewired)


        if save_computations:
            pickle.dump(speed_samples_rewired,
                        open(spreading_pickled_samples_directory_address + 'speed_samples_'
                             + str(rewiring_percentage) +
                             '_percent_rewiring_' + network_group + network_id
                             + model_id + '_' + str(theta) + '.pkl', 'wb'))
            pickle.dump(infection_size_samples_rewired,
                        open(spreading_pickled_samples_directory_address + 'infection_size_samples_'
                             + str(rewiring_percentage) +
                             '_percent_rewiring_' + network_group + network_id
                             + model_id + '_' + str(theta) + '.pkl', 'wb'))

            pickle.dump(fraction_evolution_rewired,
                        open(spreading_pickled_samples_directory_address + 'fractional_evolution_samples_'
                             + str(rewiring_percentage) +
                             '_percent_rewiring_' + network_group + network_id
                             + model_id + '_' + str(theta) + '.pkl', 'wb'))
    return


def measure_triad_addition_spread_time(network_id, percent_more_edges, theta):
    print('network id', network_id, 'triad edge addition: ', percent_more_edges, 'theta_distribution: ', theta)

    if CHECK_FOR_EXISTING_PKL_SAMPLES:
        path = Path(spreading_pickled_samples_directory_address + 'infection_size_samples_'
                    + str(percent_more_edges) + '_percent_' + 'add_triad_'
                    + network_group + network_id
                    + model_id + '_' + str(theta) + '.pkl')
        if path.is_file():
            print('infection_size_samples_' + str(percent_more_edges) + '_percent_' + 'add_triad_'
                  + network_group + network_id
                  + model_id + '_' + str(theta) + ' already exists')
            return

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
        'original_network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'triadic_closures',
        'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
        'initialization_mode': 'fixed_number_initial_infection',
        'initial_infection_number': number_initial_seeds,
        'delta': delta,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        'alpha': alpha,
        'gamma': gamma,
        # 'theta': theta,
        'theta_distribution': theta,
        'rewire': False,
    }

    if model_id == '_model_4':
        dynamics_add_triad = SimpleOnlyAlongOriginalEdges(params_add_triad)
    elif model_id == '_model_5':
        params_add_triad['relative_threshold'] = zeta
        dynamics_add_triad = RelativeLinear(params_add_triad)
    elif model_id in ['_model_1', '_model_2', '_model_3', '_model_6', '_model_7']:
        # dynamics_add_triad = DeterministicLinear(params_add_triad)
        dynamics_add_triad = ProbabilityDistributionLinear(params_add_triad)
    else:
        print('model_id is not valid')
        exit()

    speed_add_triad, std_add_triad, _, _, speed_samples_add_triad, \
        infection_size_add_triad, infection_size_std_add_triad, _, _, infection_size_samples_add_triad,fractional_evolution_add_triad = \
        dynamics_add_triad.avg_speed_of_spread(
            dataset_size=size_of_dataset,
            cap=CAP,
            mode='max')

    if VERBOSE:

        print('mean spread time in triad addition network:', speed_add_triad, std_add_triad)
        print('spread time samples in triad addition network:', speed_samples_add_triad)
        print('mean infection size in triad addition network:', infection_size_add_triad, infection_size_std_add_triad)
        print('infection size samples in triad addition network:', infection_size_samples_add_triad)
        print('fractional series in triad addition network:', fractional_evolution_add_triad)
        print(NX.is_connected(G))

    if save_computations:
        pickle.dump(speed_samples_add_triad, open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                  + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                  + network_group + network_id
                                                  + model_id + '_' + str(theta) + '.pkl', 'wb'))
        pickle.dump(infection_size_samples_add_triad, open(spreading_pickled_samples_directory_address
                                                           + 'infection_size_samples_'
                                                           + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                           + network_group + network_id
                                                           + model_id + '_' + str(theta) + '.pkl', 'wb'))
        pickle.dump(fractional_evolution_add_triad, open(spreading_pickled_samples_directory_address
                                                           + 'fractional_evolution_samples_'
                                                           + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                           + network_group + network_id
                                                           + model_id + '_' + str(theta) + '.pkl', 'wb'))

    return


def measure_random_addition_spread_time(network_id, percent_more_edges, theta):
    print('network id', network_id, 'random edge addition: ', percent_more_edges, 'theta_distribution: ', theta)

    if CHECK_FOR_EXISTING_PKL_SAMPLES:
        path = Path(spreading_pickled_samples_directory_address + 'infection_size_samples_'
                    + str(percent_more_edges) + '_percent_' + 'add_random_'
                    + network_group + network_id
                    + model_id + '_' + str(theta) + '.pkl')
        if path.is_file():
            print('infection_size_samples_' + str(percent_more_edges)
                  + '_percent_' + 'add_random_'
                  + network_group + network_id
                  + model_id + '_' + str(theta) + ' already exists')
            return

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
        'original_network': G,
        'size': network_size,
        'add_edges': True,
        'edge_addition_mode': 'random',
        'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
        'initialization_mode': 'fixed_number_initial_infection',
        'initial_infection_number': number_initial_seeds,
        'delta': delta,
        'fixed_prob_high': fixed_prob_high,
        'fixed_prob': fixed_prob_low,
        'alpha': alpha,
        'gamma': gamma,
        # 'theta': theta,
        'theta_distribution': theta,
        'rewire': False,
    }

    if model_id == '_model_4':
        dynamics_add_random = SimpleOnlyAlongOriginalEdges(params_add_random)
    elif model_id == '_model_5':
        params_add_random['relative_threshold'] = zeta
        dynamics_add_random = RelativeLinear(params_add_random)
    elif model_id in ['_model_1', '_model_2', '_model_3','_model_6', '_model_7']:
        dynamics_add_random = ProbabilityDistributionLinear(params_add_random)
        # dynamics_add_random = DeterministicLinear(params_add_random)
    else:
        print('model_id is not valid')
        exit()

    speed_add_random, std_add_random, _, _, speed_samples_add_random, \
        infection_size_add_random, infection_size_std_add_random, _, _, infection_size_samples_add_random, fractional_evolution_random = \
        dynamics_add_random.avg_speed_of_spread(
            dataset_size=size_of_dataset,
            cap=CAP,
            mode='max')

    if VERBOSE:
        print('mean spread time in add random networks: ', speed_add_random, std_add_random)
        print('spread time samples in add random networks: ', speed_samples_add_random)
        print('mean infection size in add random networks: ', infection_size_add_random, std_add_random)
        print('infection size samples in add random networks: ', infection_size_samples_add_random)
        print('fractional evolution in add random networks: ', fractional_evolution_random)

    if save_computations:
        pickle.dump(speed_samples_add_random,
                    open(spreading_pickled_samples_directory_address + 'speed_samples_'
                         + str(percent_more_edges) + '_percent_' + 'add_random_'
                         + network_group + network_id
                         + model_id + '_' + str(theta) + '.pkl', 'wb'))
        pickle.dump(infection_size_samples_add_random,
                    open(spreading_pickled_samples_directory_address + 'infection_size_samples_'
                         + str(percent_more_edges) + '_percent_' + 'add_random_'
                         + network_group + network_id
                         + model_id + '_' + str(theta) + '.pkl', 'wb'))

        pickle.dump(fractional_evolution_random,
                    open(spreading_pickled_samples_directory_address + 'fractional_evolution_samples_'
                         + str(percent_more_edges) + '_percent_' + 'add_random_'
                         + network_group + network_id
                         + model_id + '_' + str(theta) + '.pkl', 'wb'))



    return


def measure_any_intervention_spread_time(intervention_type, network_id, percent_more_edges, theta):
    if intervention_type == 'rewiring':
        measure_rewiring_spread_time(network_id, percent_more_edges, theta)
    elif intervention_type == 'triad-addition':
        measure_triad_addition_spread_time(network_id, percent_more_edges, theta)
    elif intervention_type == 'random-addition':
        measure_random_addition_spread_time(network_id, percent_more_edges, theta)


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with multiprocessing.Pool(processes = 4) as pool:
        # with multiprocessing.Pool(processes = number_CPU) as pool:
            # do computations for the original networks:
            if do_computations_for_original_network:
                pool.starmap(measure_rewiring_spread_time, product(network_id_list, [0], theta_list))
            # do computations for the modified networks:
            if intervention_type == 'rewiring':
                pool.starmap(measure_rewiring_spread_time, product(network_id_list, rewiring_percentage_list, theta_list))
                pool.close()
                pool.join()
            elif intervention_type == 'triad-addition':
                pool.starmap(measure_triad_addition_spread_time, product(network_id_list, percent_more_edges_list, theta_list))
                pool.close()
                pool.join()
            elif intervention_type == 'random-addition':
                pool.starmap(measure_random_addition_spread_time, product(network_id_list, percent_more_edges_list, theta_list))
                pool.close()
                pool.join()
            elif intervention_type == 'all':
                pool.starmap(measure_any_intervention_spread_time,
                             product(all_interventions, network_id_list, percent_more_edges_list, theta_list))
                pool.close()
                pool.join()
            elif intervention_type == 'addition':
                pool.starmap(measure_any_intervention_spread_time,
                             product(addition_interventions, network_id_list, percent_more_edges_list, theta_list))
                pool.close()
                pool.join()
            else:
                assert False, "intervention type not supported"

    else:  # no multi-processing
        # do computations for the original networks:
        if do_computations_for_original_network:
            for network_id in network_id_list:
                for theta in theta_list:
                    measure_rewiring_spread_time(network_id, 0, theta)
        # do computations for the modified networks:
        if intervention_type == 'rewiring':
            # spreading time computations for rewiring interventions
            for network_id in network_id_list:
                for rewiring_percentage in rewiring_percentage_list:
                    for theta in theta_list:
                        measure_rewiring_spread_time(network_id, rewiring_percentage, theta)
        elif intervention_type == 'triad-addition':
            # spreading time computations for triad edge addition interventions
            for network_id in network_id_list:
                for percent_more_edges in percent_more_edges_list:
                    for theta in theta_list:
                        measure_triad_addition_spread_time(network_id, percent_more_edges, theta)
        elif intervention_type == 'random-addition':
            # spreading time computations for random edge addition interventions
            for network_id in network_id_list:
                for percent_more_edges in percent_more_edges_list:
                    for theta in theta_list:
                        measure_random_addition_spread_time(network_id, percent_more_edges, theta)
        elif intervention_type == 'all':
            # spreading time computations for all interventions
            for network_id in network_id_list:
                # for theta in theta_list:
                for rewiring_percentage in rewiring_percentage_list:
                    for theta in theta_list:
                        measure_rewiring_spread_time(network_id, rewiring_percentage, theta)
                for percent_more_edges in percent_more_edges_list:
                    for theta in theta_list:
                        measure_triad_addition_spread_time(network_id, percent_more_edges, theta)
                        measure_random_addition_spread_time(network_id, percent_more_edges, theta)
        elif intervention_type == 'addition':
            # spreading time computations for all interventions
            for network_id in network_id_list:
                for percent_more_edges in percent_more_edges_list:
                    for theta in theta_list:
                        measure_triad_addition_spread_time(network_id, percent_more_edges, theta)
                        measure_random_addition_spread_time(network_id, percent_more_edges, theta)
        else:
            assert False, "intervention type not supported"