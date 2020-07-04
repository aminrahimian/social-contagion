# plotting the computed spread times in computing_spread_time_c_k_union_ER

from settings import *
from computing_spread_time_ck_union_ER import network_sizes, \
    number_of_cycle_neighbors_list, \
    size_of_dataset

if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert (simulation_type == 'ck_union_ER_vs_size') or \
           (simulation_type == 'ck_union_ER_vs_k'), \
        "we are not in the right simulation_type:" + simulation_type

    spread_time_avgs = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_avg_'
                                        + simulation_type
                                        + '.pkl', 'rb'))

    spread_time_stds = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_std_'
                                        + simulation_type
                                        + '.pkl', 'rb'))


    if simulation_type == 'ck_union_ER_vs_size':

        assert False, "the simulation type is not ready"

        # plt.figure(0)
        #
        # for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
        #     number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
        #     plt.errorbar(network_sizes,
        #                  (np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]) /
        #                   (np.asarray(network_sizes) ** (1 / 2))),
        #                  yerr=(np.asarray([1.96*s/np.sqrt(size_of_dataset) for s in
        #                        spread_time_stds[number_of_cycle_neighbors_index]])/
        #                        (np.asarray(network_sizes) ** (1 / 2))),
        #                  linewidth=1.5,
        #                  label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')
        #
        # plt.ylabel('time to spread (normalized by $\\times n^{-1/2}$)', fontsize=15)
        #
        # plt.xlabel('network size ($n$)', fontsize=15)
        # # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        # #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        # plt.legend()
        #
        # if show_plots:
        #     plt.show()
        # if save_plots:
        #     plt.savefig(theory_simulation_output_address + simulation_type + '.png')
        #
        # plt.figure(1)
        #
        # for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
        #     number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
        #     plt.plot(np.log(np.asarray(network_sizes)),
        #                  np.log((np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]))),
        #                  linewidth=1.5,
        #                  label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')
        #
        # plt.plot(np.log(np.asarray(network_sizes)),
        #          (1/2)*np.log(np.asarray(network_sizes))+0.1,
        #          linewidth=1.5,
        #          label='$(1/2)\log(n)$',  linestyle=':')
        #
        # plt.plot(np.log(np.asarray(network_sizes)),
        #          (2 / 3) * np.log(np.asarray(network_sizes))-1.45,
        #          linewidth=1.5,
        #          label='$(2/3)\log(n)$', linestyle=':')
        # #
        # # plt.plot(np.log(np.asarray(network_sizes)),
        # #          (1 / 3) * np.log(np.asarray(network_sizes)),
        # #          linewidth=1.5,
        # #          label='$(1/3)\log(n)$', linestyle=':')
        #
        # # plt.plot(np.log(np.asarray(network_sizes)),
        # #          (5 / 12) * np.log(np.asarray(network_sizes))-0.75,
        # #          linewidth=1.5,
        # #          label='$(5/12)\log(n)$', linestyle=':')
        #
        # plt.ylabel('log time to spread', fontsize=15)
        #
        # plt.xlabel('log network size ($\log(n)$)', fontsize=15)
        # # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        # #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        # plt.legend(fontsize=10)
        #
        # if show_plots:
        #     plt.show()
        # if save_plots:
        #     plt.savefig(theory_simulation_output_address + simulation_type + '_log_log_' + '.png')
        #
        # plt.figure(2)
        #
        # for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
        #     number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
        #     plt.errorbar(network_sizes,
        #                  (np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]) /
        #                   (np.asarray(network_sizes) ** (2 / 3))),
        #                  yerr=(np.asarray([1.96 * s / np.sqrt(size_of_dataset) for s in
        #                                    spread_time_stds[number_of_cycle_neighbors_index]]) /
        #                        (np.asarray(network_sizes) ** (2 / 3))),
        #                  linewidth=1.5,
        #                  label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')
        #
        # plt.ylabel('time to spread (normalized by $\\times n^{-2/3}$)', fontsize=15)
        #
        # plt.xlabel('network size ($n$)', fontsize=15)
        # # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        # #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        # plt.legend()
        #
        # if show_plots:
        #     plt.show()
        # if save_plots:
        #     plt.savefig(theory_simulation_output_address + simulation_type + '_two_third_normalization' + '.png')
        #
        # plt.figure(4)
        #
        # for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
        #     number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
        #     plt.errorbar(network_sizes,
        #                  (np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]) /
        #                   (np.asarray(network_sizes) ** (5 / 12))),
        #                  yerr=(np.asarray([1.96 * s / np.sqrt(size_of_dataset) for s in
        #                                    spread_time_stds[number_of_cycle_neighbors_index]]) /
        #                        (np.asarray(network_sizes) ** (5 / 12))),
        #                  linewidth=1.5,
        #                  label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')
        #
        # plt.ylabel('time to spread (normalized by $\\times n^{-5/12}$)', fontsize=15)
        #
        # plt.xlabel('network size ($n$)', fontsize=15)
        # # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        # #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        # plt.legend()
        #
        # if show_plots:
        #     plt.show()
        # if save_plots:
        #     plt.savefig(theory_simulation_output_address + simulation_type + '_five_twelfth_normalization' + '.png')
        #
        # plt.figure(3)
        #
        # for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
        #     number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
        #     plt.errorbar(network_sizes,
        #                  (np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]) /
        #                   (np.asarray(network_sizes) ** (1 / 3))),
        #                  yerr=(np.asarray([1.96 * s / np.sqrt(size_of_dataset) for s in
        #                                    spread_time_stds[number_of_cycle_neighbors_index]]) /
        #                        (np.asarray(network_sizes) ** (1 / 3))),
        #                  linewidth=1.5,
        #                  label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')
        #
        # plt.ylabel('time to spread (normalized by $\\times n^{-1/3}$)', fontsize=15)
        #
        # plt.xlabel('network size ($n$)', fontsize=15)
        # # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        # #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        # plt.legend()
        #
        # if show_plots:
        #     plt.show()
        # if save_plots:
        #     plt.savefig(theory_simulation_output_address + simulation_type + '_one_third_normalization' + '.png')

    elif simulation_type == 'ck_union_ER_vs_k':

        plt.figure(1, (8, 7))

        for network_size in network_sizes:
            network_size_index = network_sizes.index(network_size)
            plt.errorbar([x//2 for x in number_of_cycle_neighbors_list],
                         spread_time_avgs[network_size_index],
                         yerr=[1.96 * s / np.sqrt(size_of_dataset) for s in spread_time_stds[network_size_index]],
                         linewidth=2.5,
                         label='$n = ' + str(network_size) + '$')
        k_axis = list([x // 2 for x in number_of_cycle_neighbors_list])
        k_ticks = range(min(k_axis),max(k_axis)+1)
        plt.xticks(k_ticks,fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('time to spread', fontsize=20)
        plt.xlabel('order of cycle-power ($k$)', fontsize=20)
        # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (15 - 2k)/n$ \\end{center}', fontsize=18)
        plt.legend(prop={'size': 20})

        if show_plots:
            plt.show()
        if save_plots:
            # plt.savefig(theory_simulation_output_address + simulation_type + '.png')
            plt.savefig('./figures/' + 'figureS4.pdf')
    else:
        assert False, "simulation_type not set properly: " + simulation_type

