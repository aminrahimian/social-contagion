# plotting theory simulations for c_1 c_2 interpolation (Theorem 3).


from models import *

from computing_spread_time_c1_c2_interpolation import etas, eta_labels, q_labels, qs


if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1' \
           or 'c1_c2_interpolation_SimpleOnlyAlongC1', "we are not in the right simulation_type:" + simulation_type

    avg_spread_times = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_avg_'
                                        + simulation_type
                                        + '.pkl', 'rb'))
    std_spread_times = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_std_'
                                        + simulation_type
                                        + '.pkl', 'rb'))

    plt.figure()

    for q_label in q_labels:

        plt.errorbar(etas[:8], avg_spread_times[q_labels.index(q_label)][:8],
                     yerr=avg_spread_times[q_labels.index(q_label)][:8],
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