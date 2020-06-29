# Spread time in C_1 uninon G(n,p_n) allowing for simple adoptions with probability (q) - theorem 2 simulations.

from models import *

size_of_dataset = 500

NET_SIZE = 1000

models_list = ['Logit', 'Probit', 'Threshold']

alphas_logit = [2 / 17, 2 / 16, 3 / 16, 4 / 16, 5 / 16, 6 / 16, 7 / 16, 8 / 16, 9 / 16, 10 / 16, 11/16, 12/16]#, 8/16, 9/16,10/16]  # 7/8
alphas_logit_labels = [str(x) for x in range(len(alphas_logit))]


alphas_probit = [1 / 32, 2 / 32, 3 / 32, 4 / 32, 6 / 32, 8 / 32, 10 / 32, 12 / 32, 14 / 32, 16 / 32]
alphas_probit_labels = [str(x) for x in range(len(alphas_probit))]

alphas_mix = [3 / 16, 4 / 16, 5 / 16, 6 / 16, 7 / 16, 8 / 16, 9 / 16, 10 / 16, 11/16, 12/16]
alphas_mix_labels = [str(x) for x in range(len(alphas_mix))]


alphas = [alphas_logit, alphas_probit, alphas_mix]
alphas_labels = [alphas_logit_labels, alphas_probit_labels, alphas_mix_labels]


def compute_spread_time(model,alpha):
    if model == 'Logit':
        params_1000_logit = {
            'zero_at_zero': False,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [infected*active] + [infected*active] + [susceptible] * (NET_SIZE - 2),
            # two initial seeds, next to each other
            'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'theta': 1.5,
            'c': 2,
            'sigma': 1 / (2 * np.log(NET_SIZE ** alpha)),
            'rewire': False
        }

        dynamics = Logit(params_1000_logit)

    elif model == 'Probit':
        params_1000_probit = {
            'zero_at_zero': False,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [infected*active] + [infected*active] + [susceptible] * (NET_SIZE - 2),
            # two initial seeds, next to each other
            'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'theta': 1.5,
            'c': 2,
            'sigma': 1 / np.sqrt(8 * np.log(NET_SIZE ** alpha)),
            'rewire': False,
        }

        dynamics = Probit(params_1000_probit)

    elif model == 'Threshold':
        params_1000_mix = {
            'zero_at_zero': True,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [infected*active] + [infected*active] + [susceptible] * (NET_SIZE - 2),
            # two initial seeds, next to each other
            'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'fixed_prob_high': 1.0,
            'fixed_prob': 1 / NET_SIZE ** alpha,
            'theta': 2,
            'c': 2,
            'rewire': False,
        }

        dynamics = DeterministicLinear(params_1000_mix)

    else:
        assert False, "model is not supported for C_1 union random graph simulations"

    spread_time_avg, spread_time_std, _, _, _, _, _, _, _, _ = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    adoption_probabilities = dynamics.get_activation_probabilities()

    q = adoption_probabilities[1]

    if save_computations:
        model_index = models_list.index(model)
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + model + '_alpha_'
                                          + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                          + '_spread_time_std.pkl', 'wb'))
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + model + '_alpha_'
                                          + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                          + '_spread_time_avg.pkl', 'wb'))
        pickle.dump(q, open(theory_simulation_pickle_address
                                          + model + '_alpha_'
                                          + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                          + '_q.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations or data_dump, "we should be in do_computations or data_dump mode!"
    assert simulation_type == 'c1_union_ER', "simulation type is: " + simulation_type

    input_list = [(model, alpha) for model in models_list for alpha in alphas[models_list.index(model)]]

    if not data_dump:
        if do_multiprocessing:
            with multiprocessing.Pool(processes=number_CPU) as pool:
                pool.starmap(compute_spread_time, input_list)
            pool.close()
            pool.join()
        else:
            for model in models_list:
                for alpha in alphas[models_list.index(model)]:
                    compute_spread_time(model, alpha)
    elif data_dump:
        # load individual avg and std pkl files, organize them in a list and save them in a pair of pkl files
        # one for all the avg's and the other for all the std's.
        qs = [[] for model in models_list]
        spread_time_avgs = [[] for model in models_list]
        spread_time_stds = [[] for model in models_list]

        for model in models_list:
            model_index = models_list.index(model)
            for alpha in alphas[model_index]:
                spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + model + '_alpha_'
                                                   + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                                   + '_spread_time_avg.pkl', 'rb'))
                spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                   + model + '_alpha_'
                                                   + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                                   + '_spread_time_std.pkl', 'rb'))
                spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + model + '_alpha_'
                                                   + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                                   + '_spread_time_avg.pkl', 'rb'))
                q = pickle.load(open(theory_simulation_pickle_address
                                     + model + '_alpha_'
                                     + alphas_labels[model_index][alphas[model_index].index(alpha)]
                                     + '_q.pkl', 'rb'))

                qs[model_index].append(q)

                spread_time_avgs[model_index].append(spread_time_avg)

                spread_time_stds[model_index].append(spread_time_avg)

        pickle.dump(spread_time_stds, open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_times_std.pkl', 'wb'))

        pickle.dump(spread_time_avgs, open(theory_simulation_pickle_address
                                           + simulation_type
                                           + '_spread_time_avgs.pkl', 'wb'))

        pickle.dump(qs, open(theory_simulation_pickle_address
                             + simulation_type
                             + '_qs.pkl', 'wb'))
