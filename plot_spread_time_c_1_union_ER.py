# plotting the computed spread times in computing_spread_time_c_1_union_ER

from settings import *
from computing_spread_time_c_1_union_ER import models_list, size_of_dataset, NET_SIZE

models_list = ['logit', 'probit', 'noisy 2-complex'] # renaming labels

if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert simulation_type == 'c1_union_ER', \
        "we are not in the right simulation_type:" + simulation_type

    spread_time_stds = pickle.load(open(theory_simulation_pickle_address
                                        + simulation_type
                                        + '_spread_times_std.pkl', 'rb'))

    spread_time_avgs = pickle.load(open(theory_simulation_pickle_address
                                        + simulation_type
                                        + '_spread_time_avgs.pkl', 'rb'))

    qs = pickle.load(open(theory_simulation_pickle_address
                          + simulation_type
                          + '_qs.pkl', 'rb'))

    plt.figure(1,(7, 6))

    for model in models_list:
        model_index = models_list.index(model)
        plt.errorbar(qs[model_index],
                     spread_time_avgs[model_index],
                     yerr=[1.96*s/np.sqrt(size_of_dataset) for s in spread_time_stds[model_index]],
                     linewidth=1.5,
                     label=model)  # $\\sigma_n = 1/2\log(n^{\\alpha})$

    plt.plot(qs[models_list.index('noisy 2-complex')],
             [500]*len(qs[models_list.index('noisy 2-complex')]),
             color='r', linewidth=1, linestyle='--',
             label='$\mathcal{C}_2$ benchmark')


    plt.xscale("log")
    plt.ylabel('time to spread', fontsize=20)
    plt.xlabel('probability of adoptions below threshold $(q)$', fontsize=20)
    # plt.title('\centering Complex Contagion over $\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n},n = '
    #           + str(NET_SIZE) + '$'
    #           '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions \\end{center}', fontsize=18)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.arrow(0.0074, 500, 0, -500, width=0.0001, head_width=0.0006,
              head_length=50, capstyle='projecting',
              length_includes_head=True, shape='full', joinstyle='round', fc='k', ec='k')
    plt.text(0.008, 10, '$0.0074$',
             fontweight='bold', fontdict=None, withdash=False)
    if show_plots:
        plt.show()
    if save_plots:
        # plt.savefig(theory_simulation_output_address + simulation_type + '.png')
        plt.savefig('./figures/' + 'figureS6B.pdf')


# # plotting the computed spread times in computing_spread_time_Z2_union_ER
#
# from settings import *
# from computing_spread_time_c_1_union_ER import models_list, size_of_dataset, NET_SIZE
#
# models_list = ['noisy 2-complex'] # renaming labels
# models_list1 = ['Z4 benchmark']
#
# if __name__ == '__main__':
#
#     assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"
#
#     assert simulation_type == 'Z2_union_ER', \
#         "we are not in the right simulation_type:" + simulation_type
#
#     spread_time_stds = pickle.load(open(theory_simulation_pickle_address
#                                         + 'Z2_union_ER/'
#                                         + simulation_type
#                                         + '_spread_times_std.pkl', 'rb'))
#
#     spread_time_avgs = pickle.load(open(theory_simulation_pickle_address
#                                         + 'Z2_union_ER/'
#                                         + simulation_type
#                                         + '_spread_time_avgs.pkl', 'rb'))
#
#     qs = pickle.load(open(theory_simulation_pickle_address
#                           + 'Z2_union_ER/'
#                           + simulation_type
#                           + '_qs.pkl', 'rb'))
#
#     plt.figure(1,(7, 6))
#     spread_time_avgs_benchmark = pickle.load(open(theory_simulation_pickle_address
#                                         + 'Z4/'
#                                         + simulation_type
#                                         + '_spread_time_avgs.pkl', 'rb'))
#     spread_time_stds_benchmark = pickle.load(open(theory_simulation_pickle_address
#                                         + 'Z4/'
#                                         + simulation_type
#                                         + '_spread_times_std.pkl', 'rb'))
#     qs_benchmark = pickle.load(open(theory_simulation_pickle_address
#                           + 'Z4/'
#                           + simulation_type
#                           + '_qs.pkl', 'rb'))
#
#     for model in models_list:
#         model_index = models_list.index(model)
#         plt.errorbar(qs[model_index],
#                      spread_time_avgs[model_index],
#                      yerr=[1.96*s/np.sqrt(size_of_dataset) for s in spread_time_stds[model_index]],
#                      linewidth=1.5,
#                      label=model)
#
#     for model in models_list1:
#         model_index = models_list1.index(model)
#         plt.errorbar(qs_benchmark[model_index],
#                      spread_time_avgs_benchmark[model_index],
#                      yerr=[1.96 * s / np.sqrt(size_of_dataset) for s in spread_time_stds_benchmark[model_index]],
#                      linewidth=1.5,
#                      color='r', linestyle='--',
#                      label = '$\mathcal{Z}_4$ benchmark')
#
#
#     plt.xscale("log")
#     plt.ylabel('time to spread', fontsize=20)
#     plt.xlabel('probability of adoptions below threshold $(q)$', fontsize=20)
#     # plt.title('\centering Complex Contagion over $\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n},n = '
#     #           + str(NET_SIZE) + '$'
#     #           '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions \\end{center}', fontsize=18)
#     plt.legend(loc='upper right', prop={'size': 20})
#     plt.arrow(0.01105, 90, 0, -90, width=0.0001, head_width=0.001,
#               head_length=15, capstyle='projecting',
#               length_includes_head=True, shape='full', joinstyle='round', fc='k', ec='k')
#     plt.text(0.012, 30, '$0.01105$',
#              fontweight='bold', fontdict=None, withdash=False)
#     if show_plots:
#         plt.show()
#     if save_plots:
#         # plt.savefig(theory_simulation_output_address + simulation_type + '.png')
#         plt.savefig('./figures/' + 'Z2_union_ER_vs_q.pdf')

