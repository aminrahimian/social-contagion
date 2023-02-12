# Spread time in C_1 uninon G(n,p_n) allowing for simple adoptions with probability (q) - theorem 2 simulations.

from models import *

size_of_dataset = 2

NET_SIZE = 10

models_list = ['Threshold']
model = 'Threshold'

fixed_prob_low = [0.025, 0.05, 0.075]


def compute_spread_time(model, Q):

    if model == 'Threshold':
        params_1000_mix = {
            'zero_at_zero': True,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [infected*active] + [infected*active] + [susceptible] * (NET_SIZE - 2),
            # two initial seeds, next to each other
            'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'fixed_prob_high': 1.0,
            'fixed_prob': Q,
            'theta': 2,
            'c': 2,
            'rewire': False,
        }

        dynamics = DeterministicLinear(params_1000_mix)

    else:
        assert False, "model is not supported for C_1 union random graph simulations"

    spread_time_avg, spread_time_std, _, _, _, _, _, _, _, _, _ = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    adoption_probabilities = dynamics.get_activation_probabilities()

    q = adoption_probabilities[1]

    if save_computations:
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + model + '_Q_'
                                          + str(fixed_prob_low.index(Q))
                                          + '_spread_time_std.pkl', 'wb'))
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + model + '_Q_'
                                          + str(fixed_prob_low.index(Q))
                                          + '_spread_time_avg.pkl', 'wb'))
        pickle.dump(q, open(theory_simulation_pickle_address
                                          + model + '_Q_'
                                          + str(fixed_prob_low.index(Q))
                                          + '_q.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations or data_dump, "we should be in do_computations or data_dump mode!"
    assert simulation_type == 'c1_union_ER', "simulation type is: " + simulation_type

    input_list = [(model, Q) for model in models_list for Q in fixed_prob_low]

    if not data_dump:
        if do_multiprocessing:
            with multiprocessing.Pool(processes=number_CPU) as pool:
                pool.starmap(compute_spread_time, input_list)
            pool.close()
            pool.join()
        else:
            for model in models_list:
                for Q in fixed_prob_low[models_list.index(model)]:
                    compute_spread_time(model, Q)
    elif data_dump:
        # load individual avg and std pkl files, organize them in a list and save them in a pair of pkl files
        # one for all the avg's and the other for all the std's.
        qs = []
        spread_time_avgs = []
        spread_time_stds = []


        for Q in fixed_prob_low:
            spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + model + '_Q_'
                                                   + str(fixed_prob_low.index(Q))
                                                   + '_spread_time_avg.pkl', 'rb'))

            spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                   + model + '_Q_'
                                                   + str(fixed_prob_low.index(Q))
                                                   + '_spread_time_std.pkl', 'rb'))
            q = pickle.load(open(theory_simulation_pickle_address
                                     + model + '_Q_'
                                     + str(fixed_prob_low.index(Q))
                                     + '_q.pkl', 'rb'))

            qs.append(q)

            spread_time_avgs.append(spread_time_avg)

            spread_time_stds.append(spread_time_avg)

        pickle.dump(spread_time_stds, open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_times_std.pkl', 'wb'))

        pickle.dump(spread_time_avgs, open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_time_avgs.pkl', 'wb'))

        pickle.dump(qs, open(theory_simulation_pickle_address
                             + simulation_type
                             + '_qs.pkl', 'wb'))

        spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_times_std.pkl', 'rb'))
        print(spread_time_avg)
        spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_time_avgs.pkl', 'rb'))
        print(spread_time_std)
        qs = pickle.load(open(theory_simulation_pickle_address
                             + simulation_type
                             + '_qs.pkl', 'rb'))
        print(qs)