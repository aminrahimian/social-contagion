# Computing the spread times in C_k union G_n,pn model. We test the results of theorem 1.
# We use the (0,1) threshold model for complex contagion with theta = 2,
# if two or more neighbors are infected then the agent gets infected, otherwise not.
# The expected degree is fixed and the number of cycle edges (k)
# and expected number of random edges per node (c) is varied.


from models import *

size_of_dataset = 100

number_of_initial_infected_nodes = 2

if simulation_type == 'ck_union_ER_vs_size':

    expected_degree = 13

    network_sizes = [2000, 3000, 4000, 5000, 6000, 7000]

    number_of_cycle_neighbors_list = [12, 10, 8]


elif simulation_type == 'ck_union_ER_vs_k':

    expected_degree = 15

    number_of_cycle_neighbors_list = [14, 12, 10, 8, 6]

    network_sizes = [1000, 3000, 5000]

else:
    print("simulation_type not set properly: " + simulation_type)
    exit()

number_CPU = len(network_sizes)*len(number_of_cycle_neighbors_list)


def compute_spread_time_for_c_k(expected_degree, number_of_cycle_neighbors, network_size):
    params = {
        'network_model': 'cycle_union_Erdos_Renyi',
        'size': network_size,  # populationSize,
        'initial_states': [infected*active]*number_of_initial_infected_nodes +
                          [susceptible] * (network_size - number_of_initial_infected_nodes),
        'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'nearest_neighbors': number_of_cycle_neighbors,
        'fixed_prob_high': 1.0,
        'fixed_prob': 0.0,
        'theta': 2,
        'c': expected_degree - number_of_cycle_neighbors,
        'rewire': False,
    }

    dynamics = DeterministicLinear(params)

    spread_time_avg, spread_time_std, _, _, _, _, _, _, _, _ = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    print('network_size ', network_size, 'expected_degree ', expected_degree,
          'number_of_cycle_edges', number_of_cycle_neighbors, 'spread_time_avg',
          spread_time_avg, 'spread_time_std', spread_time_std)

    if save_computations:
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + 'spreading_time_avg'
                                          + '_D_' + str(expected_degree) + '_k_'
                                          + str(number_of_cycle_neighbors) + '_net_size_'
                                          + str(network_size)
                                          + '.pkl', 'wb'))
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + 'spreading_time_std'
                                          + '_D_' + str(expected_degree) + '_k_'
                                          + str(number_of_cycle_neighbors) + '_net_size_'
                                          + str(network_size)
                                          + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode!"

    assert (simulation_type == 'ck_union_ER_vs_size') or \
           (simulation_type == 'ck_union_ER_vs_k'), \
        "simulation_type not set properly: " + simulation_type

    if do_multiprocessing:
        with multiprocessing.Pool(processes=number_CPU) as pool:
            pool.starmap(compute_spread_time_for_c_k, product([expected_degree],
                                                              number_of_cycle_neighbors_list,
                                                              network_sizes))
            pool.close()
            pool.join()
    else:  # no multi-processing:
        for network_size in network_sizes:
            for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
                compute_spread_time_for_c_k(expected_degree,
                                            number_of_cycle_neighbors,
                                            network_size)

    if save_computations:
        avg_spread_times = []
        std_spread_times = []
        if simulation_type == 'ck_union_ER_vs_size':
            for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
                spread_avg = []
                spread_std = []
                for network_size in network_sizes:
                    spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                       + 'spreading_time_avg'
                                                       + '_D_' + str(expected_degree) + '_k_'
                                                       + str(number_of_cycle_neighbors) + '_net_size_'
                                                       + str(network_size)
                                                       + '.pkl', 'rb'))
                    spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                       + 'spreading_time_std'
                                                       + '_D_' + str(expected_degree) + '_k_'
                                                       + str(number_of_cycle_neighbors) + '_net_size_'
                                                       + str(network_size)
                                                       + '.pkl', 'rb'))

                    spread_avg.append(spread_time_avg)
                    spread_std.append(spread_time_std)

                print(spread_avg)
                print(spread_std)

                avg_spread_times.append(spread_avg)

                std_spread_times.append(spread_std)

            print(avg_spread_times)
            print(std_spread_times)

            pickle.dump(avg_spread_times, open(theory_simulation_pickle_address
                                               + 'spreading_time_avg_'
                                               + simulation_type
                                               + '.pkl', 'wb'))
            pickle.dump(std_spread_times, open(theory_simulation_pickle_address
                                               + 'spreading_time_std_'
                                               + simulation_type
                                               + '.pkl', 'wb'))
        elif simulation_type == 'ck_union_ER_vs_k':
            for network_size in network_sizes:
                spread_avg = []
                spread_std = []
                for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
                    spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                       + 'spreading_time_avg'
                                                       + '_D_' + str(expected_degree) + '_k_'
                                                       + str(number_of_cycle_neighbors) + '_net_size_'
                                                       + str(network_size)
                                                       + '.pkl', 'rb'))
                    spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                       + 'spreading_time_std'
                                                       + '_D_' + str(expected_degree) + '_k_'
                                                       + str(number_of_cycle_neighbors) + '_net_size_'
                                                       + str(network_size)
                                                       + '.pkl', 'rb'))

                    spread_avg.append(spread_time_avg)
                    spread_std.append(spread_time_std)

                print(spread_avg)
                print(spread_std)

                avg_spread_times.append(spread_avg)

                std_spread_times.append(spread_std)

            print(avg_spread_times)
            print(std_spread_times)

            pickle.dump(avg_spread_times, open(theory_simulation_pickle_address
                                               + 'spreading_time_avg_'
                                               + simulation_type
                                               + '.pkl', 'wb'))
            pickle.dump(std_spread_times, open(theory_simulation_pickle_address
                                               + 'spreading_time_std_'
                                               + simulation_type
                                               + '.pkl', 'wb'))

