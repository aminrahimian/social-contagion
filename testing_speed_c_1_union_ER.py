# Comparing the speed in C_k uninon G(n,p_n). We use the (0,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent
# does not get infected.

from models import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

RD.seed()

size_of_dataset = 20


if __name__ == '__main__':

    alphas = [1 / 16, 2 / 16, 3 / 16, 4 / 16, 5 / 16, 6 / 16, 7/16, 8/16, 9/16,10/16]  # 7/8

    NET_SIZE = 1000

    speeds_1000_logit = []
    speeds_std_1000_logit = []

    for alpha in alphas:

        params_1000_logit = {
            'zero_at_zero': False,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [1] + [1] + [0] * (NET_SIZE - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'theta': 1.5,
            'c': 1,
            'sigma': 1/(2*np.log(NET_SIZE**alpha)),
        }

        print(params_1000_logit['size'])
        dynamics = Logit(params_1000_logit)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_1000_logit.append(speed)
        speeds_std_1000_logit.append(std)
        print(speeds_1000_logit)
        print(speeds_std_1000_logit)
        print(dynamics.get_activation_probabilities())

    speeds_1000_probit = []
    speeds_std_1000_probit = []

    for alpha in alphas:
        params_1000_probit = {
            'zero_at_zero': False,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [1] + [1] + [0] * (NET_SIZE - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'theta': 1.5,
            'c': 1,
            'sigma': 1 / np.sqrt(8 * np.log(NET_SIZE ** alpha)),
        }

        print(params_1000_probit['size'])
        dynamics = Probit(params_1000_probit)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_1000_probit.append(speed)
        speeds_std_1000_probit.append(std)
        print(speeds_1000_probit)
        print(speeds_std_1000_probit)
        print(dynamics.get_activation_probabilities())

    speeds_1000_mix = []
    speeds_std_1000_mix = []

    for alpha in alphas:
        params_1000_mix = {
            'zero_at_zero': True,
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': NET_SIZE,  # populationSize,
            'initial_states': [1] + [1] + [0] * (NET_SIZE - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'fixed_prob_high': 1.0,
            'fixed_prob': 1 / NET_SIZE ** alpha,
            'theta': 2,
            'c': 1,
        }
        print(params_1000_mix['size'])

        dynamics = DeterministicLinear(params_1000_mix)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_1000_mix.append(speed)
        speeds_std_1000_mix.append(std)
        print(speeds_1000_mix)
        print(speeds_std_1000_mix)

        print(dynamics.get_activation_probabilities())

    # NET_SIZE = 3000
    #
    # speeds_3000 = []
    # speeds_std_3000 = []
    #
    # for alpha in alphas:
    #     params_3000 = {
    #         'network_model': 'cycle_union_Erdos_Renyi',
    #         'size': NET_SIZE,  # populationSize,
    #         'initial_states': [1] + [1] + [0] * (NET_SIZE - 2),  # two initial seeds, next to each other
    #         'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
    #         'nearest_neighbors': 2,
    #         'theta': 2,
    #         'c': 1,
    #         'sigma': 1 / np.log(NET_SIZE**alpha),
    #     }
    #     print(params_3000['size'])
    #     dynamics = logit(params_3000)
    #     speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
    #     speeds_3000.append(speed)
    #     speeds_std_3000.append(std)
    #     print(speeds_3000)
    #     print(speeds_std_3000)
    #
    # NET_SIZE = 5000
    #
    # speeds_5000 = []
    # speeds_std_5000 = []
    #
    # for alpha in alphas:
    #     params_5000 = {
    #         'network_model': 'cycle_union_Erdos_Renyi',
    #         'size': NET_SIZE,  # populationSize,
    #         'initial_states': [1] + [1] + [0] * (NET_SIZE - 2),  # two initial seeds, next to each other
    #         'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
    #         'nearest_neighbors': 2,
    #         'theta': 2,
    #         'c': 1,
    #         'sigma': 1 / np.log(NET_SIZE**alpha),
    #     }
    #     print(params_5000['size'])
    #     dynamics = logit(params_5000)
    #     speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
    #     speeds_5000.append(speed)
    #     speeds_std_5000.append(std)
    #     print(speeds_5000)
    #     print(speeds_std_5000)

    plt.figure(2)
    #
    if 'speeds_1000_logit' in vars() or 'speeds_1000_logit' in globals():
        plt.errorbar(alphas, np.asarray(speeds_1000_logit), yerr=speeds_std_1000_logit, color='g', linewidth=1.5,
                     label='Logit $\\sigma_n = 1/2\log(n^{\\alpha})$')

    if 'speeds_1000_probit' in vars() or 'speeds_1000_probit' in globals():
        plt.errorbar(alphas, np.asarray(speeds_1000_probit), yerr=speeds_std_1000_probit, color='m', linewidth=1.5,
                     label='Probit $\\sigma_n = 1/\sqrt{8\log(n^{\\alpha})}$')

    plt.plot(alphas, [500]*len(alphas), color='c', linewidth=1,
                 label='The $\mathcal{C}_2$ benchmark')

    if 'speeds_1000_mix' in vars() or 'speeds_1000_mix' in globals():
        plt.errorbar(alphas, np.asarray(speeds_1000_mix), yerr=speeds_std_1000_mix, color='r', linewidth=2.5,
                     label='Linear Threshold $q_n = 1/n^{\\alpha}$')

    # plt.errorbar(alphas, speeds_3000, yerr=speeds_std_3000, color='g', linewidth=2.5,
    #              label='$n = 3000$')
    #
    # plt.errorbar(alphas, speeds_5000, yerr=speeds_std_5000, color='c', linewidth=2.5,
    #              label='$n = 5000$')

    plt.ylabel('Time to Spread')
    plt.xlabel('$\\alpha$')
    plt.title('\centering Complex Contagion with Sub-threshold Adoptions '
              '\\vspace{-10pt}  \\begin{center} over   $\mathcal{C}_{1} \\cup \mathcal{G}_{n,1/n}, n = 1000$ \\end{center}')
    plt.legend()
    # plt.show()
    plt.savefig('./data/' + 'speed_of_contagion_union_C_1.png')