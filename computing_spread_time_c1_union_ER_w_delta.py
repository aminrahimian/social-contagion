# Computing the spread times in C_1 union ER model. We use the (q,1) threshold model
# for complex contagion with theta = 2,
# we test the effect of delta> 0

from models import *

size_of_dataset = 1000

network_size = 500

deltas = list(np.linspace(0, .1, 50))
delta_labels = [str(x) for x in range(len(deltas))]
print(deltas)
print(delta_labels)

qs = [0.0447, 0.0833, 0.1550, 0.2115, 0.2885]
q_labels = [str(x) for x in range(len(qs))]
print(qs)
print(q_labels)

def compute_spread_time_for_q_delta(q, delta):
    params = {
        'zero_at_zero': True,
        'network_model': 'cycle_union_Erdos_Renyi',
        'size': network_size,
        'initial_states': [infected*active] + [infected*active] + [susceptible] * (network_size - 2),  # two initial seeds, next to each other
        'delta': delta,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'fixed_prob_high': 1.0,
        'fixed_prob': q,
        'theta': 2,
        'c': 2,
        'nearest_neighbors': 2,
        'rewire': False,
    }

    print('delta: ', params['delta'], 'q: ', params['fixed_prob'])

    if simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1' or 'c1_union_ER_with_delta':
        dynamics = SimpleOnlyAlongC1(params)
    elif simulation_type is 'c1_union_ER':
        dynamics = DeterministicLinear(params)

    spread_time_avg, spread_time_std, _, _, samples, _, _, _, _, _ = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    print('spread_time_avg: ', spread_time_avg)
    print('spread_time_std: ', spread_time_std)

    if save_computations:
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + 'spreading_time_avg'
                                          + '_delta_' + delta_labels[deltas.index(delta)] + '_q_'
                                          + '_q_' + q_labels[qs.index(q)]
                                          + '.pkl', 'wb'))
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + 'spreading_time_std'
                                          + '_delta_' + delta_labels[deltas.index(delta)]
                                          + '_q_' + q_labels[qs.index(q)]
                                          + '.pkl', 'wb'))
    return spread_time_avg, spread_time_std


if __name__ == '__main__':
    assert do_computations or data_dump, "we should be in do_computations or data_dump mode!"

    assert simulation_type == 'c1_union_ER_with_delta', "simulation type is: " + simulation_type

    if not data_dump:
        if do_multiprocessing:
            with multiprocessing.Pool(processes=number_CPU) as pool:
                pool.starmap(compute_spread_time_for_q_delta, product(qs, deltas))
                pool.close()
                pool.join()
        else:  # no multi-processing:
            for q in qs:
                for delta in deltas:
                    compute_spread_time_for_q_delta(q, delta)

    elif data_dump:
        # load individual avg and std pkl files, organize them in a list and save them in a pair of pkl files
        # one for all the avg's and the other for all the std's.
        avg_spread_times = []
        std_spread_times = []
        for q in qs:
            spread_avg = []
            spread_std = []
            for delta in deltas:
                spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_avg'
                                                   + '_delta_' + delta_labels[deltas.index(delta)] + '_q_'
                                                   + '_q_' + q_labels[qs.index(q)]
                                                   + '.pkl', 'rb'))
                spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_std'
                                                   + '_delta_' + delta_labels[deltas.index(delta)]
                                                   + '_q_' + q_labels[qs.index(q)]
                                                   + '.pkl', 'rb'))
                # spread_time_avg, spread_time_std = compute_spread_time_for_q_delta(q, delta)
                spread_avg.append(spread_time_avg)
                spread_std.append(spread_time_std)
            print(spread_time_avg)
            print(spread_time_std)
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
