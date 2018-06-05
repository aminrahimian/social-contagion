# Comparing the speed in C_k uninon G(n.p_n). We use the (0,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent
# does not get infected.

from models import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

RD.seed()

size_of_dataset = 10


if __name__ == '__main__':

    network_sizes = [1000, 2000, 3000, 4000]#[10]#list(np.int_(np.ceil((np.logspace(2.0, 5.0, num=10)))))
#[1000, 2000, 3000, 4000]#,5000,6000,7000,8000,9000,10000]#, 2000, 3000, 4000]
    speeds_12_1 = []
    speeds_8_5 = []
    speeds_4_9 = []

    for network_size in network_sizes:
        params_12_1 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size-2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 12,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 1,
        }
        print(params_12_1['size'])
        dynamics = DeterministicLinear(params_12_1)
        speed = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_12_1.append(speed)
        print(speeds_12_1)

        params_8_5 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 8,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 5,
        }
        print(params_8_5['size'])
        dynamics = DeterministicLinear(params_8_5)
        speed = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_8_5.append(speed)
        print(speeds_8_5)

        params_4_9 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 4,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 9,
        }
        print(params_4_9['size'])
        dynamics = DeterministicLinear(params_4_9)
        speed = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_4_9.append(speed)
        print(speeds_4_9)



    plt.plot(network_sizes, speeds_12_1, color = 'r', linewidth=2.5, label='Speed scale $C_6 \\cup G_{n,1/n}$')
    plt.plot(network_sizes, speeds_8_5, color='g', linewidth=2.5, label='Speed scale $C_4 \\cup G_{n,5/n}$')
    plt.plot(network_sizes, speeds_4_9, color = 'c', linewidth=2.5, label='Speed scale $C_2 \\cup G_{n,9/n}$')

    plt.ylabel('Time to Spread')
    plt.xlabel('Network Size')
    plt.title('Speed of Complex Contagion')
    plt.legend()
    # plt.show()
    plt.savefig('./data/' + 'speed_of_contagion_union.png')