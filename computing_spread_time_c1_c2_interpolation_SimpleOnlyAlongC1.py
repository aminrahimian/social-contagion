# Comparing the speed in C_k uninon G(n,p_n). We use the (q,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If one neighbor is infected then the agent
# gets infected with probability q.


from models import *

assert do_computations, "we should be in do_computations mode!"

assert simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1' or 'c1_c2_interpolation', \
    "simulation_type not set properly: " + simulation_type

size_of_dataset = 500

network_size = 200

etas = list(np.linspace(0, 80, 10))

qs = [1 / network_size ** x for x in [0.1, 0.6, 0.8]]

q_labels = [str(x) for x in range(len(qs))]

eta_labels = [str(x) for x in range(len(etas))]

add_long_ties_exp = np.random.exponential(scale=network_size ** 2,
                                          size=int(1.0 * network_size * (network_size - 1)) // 2)
remove_cycle_edges_exp = np.random.exponential(scale=2 * network_size,

                                               size=network_size)
def compute_spread_time_for_q_eta(q, eta):
    params = {
        'zero_at_zero': True,
        'network_model': 'c_1_c_2_interpolation',
        'size': network_size,
        'initial_states': [infected*active] + [infected*active] + [susceptible] * (network_size - 2),  # two initial seeds, next to each other
        'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'fixed_prob_high': 1.0,
        'fixed_prob': q,
        'theta': 2,
        'eta': eta,
        'rewire': False,
        'add_long_ties_exp': add_long_ties_exp,
        'remove_cycle_edges_exp': remove_cycle_edges_exp,
    }

    print('eta: ', params['eta'], 'q: ', params['fixed_prob'])

    if simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1':
        dynamics = SimpleOnlyAlongC1(params)
    elif simulation_type is 'c1_c2_interpolation':
        dynamics = DeterministicLinear(params)

    spread_time_avg, spread_time_std, _, _, samples = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    print('spread_time_avg: ', spread_time_avg)
    print('spread_time_std: ', spread_time_std)

    if save_computations:
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + 'spreading_time_avg'
                                          + '_eta_' + eta_labels[etas.index(eta)] + '_q_'
                                          + q_labels[qs.index(q)]
                                          + '.pkl', 'wb'))
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + 'spreading_time_std'
                                          + '_eta_' + eta_labels[etas.index(eta)]
                                          + '_q_' + q_labels[qs.index(q)]
                                          + '.pkl', 'wb'))


if __name__ == '__main__':

    if do_multiprocessing:
        with multiprocessing.Pool(processes=number_CPU) as pool:
            # do computations for the original networks:
            pool.starmap(compute_spread_time_for_q_eta, product(qs,etas))
    else:  # no multi-processing:
        for q in qs:
            for eta in etas:
                compute_spread_time_for_q_eta(q, eta)

    if save_computations:
        for q in qs:
            spread_avg = []
            spread_std = []
            for eta in etas:
                spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_avg'
                                                   + '_eta_' + eta_labels[etas.index(eta)]
                                                   + '_q_' + q_labels[qs.index(q)]
                                                   + '.pkl', 'rb'))
                spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_std'
                                                   + '_eta_' + eta_labels[etas.index(eta)]
                                                   + '_q_' + q_labels[qs.index(q)]
                                                   + '.pkl', 'rb'))

                spread_avg.append(spread_time_avg)
                spread_std.append(spread_time_std)

            print(spread_avg)
            print(spread_std)

            pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                              + simulation_type + '_avg_points'
                                              + '.pkl', 'wb'))
            pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                              + simulation_type + '_std_points'
                                              + '.pkl', 'wb'))
