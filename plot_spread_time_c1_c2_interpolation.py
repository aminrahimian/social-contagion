# plotting theory simulations for c_1 c_2 interpolation (Theorem 3).


from models import *

from computing_spread_time_c1_c2_interpolation import etas, eta_labels, q_labels, qs


if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1' \
           or 'c1_c2_interpolation_SimpleOnlyAlongC1', "we are not in the right simulation_type:" + simulation_type

    avg_spread_times = []
    std_spread_times = []
    for q_label in q_labels:
        spread_avg = []
        spread_std = []
        for eta in etas:
            spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                               + 'spreading_time_avg'
                                               + '_eta_' + eta_labels[etas.index(eta)]
                                               + '_q_' + q_label
                                               + '.pkl', 'rb'))
            spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                               + 'spreading_time_std'
                                               + '_eta_' + eta_labels[etas.index(eta)]
                                               + '_q_' + q_label
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
                                       + 'spreading_time_avg'
                                       + simulation_type
                                       + '.pkl', 'wb'))
    pickle.dump(std_spread_times, open(theory_simulation_pickle_address
                                       + 'spreading_time_std'
                                       + simulation_type
                                       + '.pkl', 'wb'))

    plt.figure()

    for q_label in q_labels:

        plt.errorbar(etas, avg_spread_times[q_labels.index(q_label)], yerr=avg_spread_times[q_labels.index(q_label)],
                     linewidth=2.5, label='$q_n = '+str(Decimal(qs[q_labels.index(q_label)]).quantize(FOURPLACES))+'$')


    plt.ylabel('Time to Spread')
    plt.xlabel('Edges Rewired ($\\eta$)')
    plt.title('\centering Complex Contagion over $\mathcal{C}_2^{\eta}, n = 200$'
              '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions $(q_n)$ '
              'and Rewiring $(\eta)$   \\end{center}')
    plt.legend()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(theory_simulation_output_address + simulation_type + '.png')