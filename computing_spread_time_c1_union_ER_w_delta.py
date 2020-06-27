# Computing the spread times in C_2^eta model. We use the (q,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If one neighbor is infected then the agent
# gets infected with probability q. eta determines how much of the C_2\C_1 edges are rewired.


from models import *

#was 1000
size_of_dataset = 25
#was 500
network_size = 30

#originally 50 etas
deltas = list(np.linspace(.02, .1, 5))  # list(np.linspace(0, 80, 10))
print(deltas)

# these are previously collected samples
#  their pkl files already exist:
qs_old = [1 / network_size ** x for x in [0.06246936830415, 0.0352905371428536, 0.0111381973555761]]
qs = [1 / network_size ** x for x in [0.173393743, 0.129818655252878, 0.0935433216785722]]

#qs_new = [1 / network_size ** x for x in [0.06246936830415]]
#qs_new_new_old = [1 / network_size ** x for x in [0.0935433216785722]]
# these are new samples to be computed:
#qs_new_new = [1 / network_size ** x for x in [0.129818655252878]]
all_qs = qs_old[::-1] + qs[::-1]
#+ qs_new[::-1]+ qs_new_new_old[::-1] + qs_new_new[::-1]

q_labels_old = ['00'+str(x) for x in range(len(qs_old))]
q_labels = [str(x) for x in range(len(qs))]
#q_labels_new = ['000'+str(x) for x in range(len(qs_new))]
#q_labels_new_new_old = ['0000'+str(x) for x in range(len(qs_new_new_old))]
#q_labels_new_new = ['00000'+str(x) for x in range(len(qs_new_new))]
all_q_labels = q_labels_old[::-1] + q_labels[::-1] \
#+ q_labels_new[::-1] + q_labels_new_new_old[::-1] + q_labels_new_new[::-1]

#test qs
#qs1 = [1 / network_size ** x for x in [0.001]]
#qs2 = [1 / network_size ** x for x in [0.002]]
#qs3 = [1 / network_size ** x for x in [0.003]]
#all_qs = qs1[::-1] + qs2[::-1] + qs3[::-1]
#q_labels1 = ['00'+str(x) for x in range(len(qs1))]
#q_labels2 = ['000'+str(x) for x in range(len(qs2))]
#q_labels3 = ['0000'+str(x) for x in range(len(qs3))]
#all_q_labels = q_labels1[::-1] + q_labels2[::-1] + q_labels3[::-1]
#tp added this line
print(all_qs)
print(all_q_labels)

delta_labels = [str(x) for x in range(len(deltas))]

supply_the_exponentials = False

if supply_the_exponentials:
    add_long_ties_exp = np.random.exponential(scale=network_size ** 2,
                                              size=int(1.0 * network_size * (network_size - 1)) // 2)
    remove_cycle_edges_exp = np.random.exponential(scale=2 * network_size, size=network_size)


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

    if simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1':
        dynamics = SimpleOnlyAlongC1(params)
        #should this be a different setting?
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
                                          + '_q_' + all_q_labels[all_qs.index(q)]
                                          + '.pkl', 'wb'))
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + 'spreading_time_std'
                                          + '_delta_' + delta_labels[deltas.index(delta)]
                                          + '_q_' + all_q_labels[all_qs.index(q)]
                                          + '.pkl', 'wb'))
    return spread_time_avg, spread_time_std


if __name__ == '__main__':
    assert do_computations, "we should be in do_computations mode!"

    assert simulation_type == 'c1_union_ER', "simulation type is: " + simulation_type

    # if do_multiprocessing:
    #     with multiprocessing.Pool(processes=number_CPU) as pool:
    #         pool.starmap(compute_spread_time_for_q_delta, product(qs_new_new, deltas))
    #         pool.close()
    #         pool.join()
    # else:  # no multi-processing:
    #     for q in qs_new_new:
    #         for delta in deltas:
    #             compute_spread_time_for_q_delta(q, delta)

    if save_computations:
        avg_spread_times = []
        std_spread_times = []
        for q in all_qs:
            spread_avg = []
            spread_std = []
            for delta in deltas:
                spread_time_avg, spread_time_std = compute_spread_time_for_q_delta(q, delta)
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
