# plots simple, complex and noisy threshold activation functions


from models import *

degree_range = np.linspace(0,5,300)
degree_range_int = np.linspace(0,5,6)

if __name__ == '__main__':

    assert do_plots, "we should be in do_plots mode!"

    params_SI_beta_4 = {
        'beta':0.4,
        'theta': 1.5,
        'delta': 0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'size':10,
    }
    params_SI_beta_5 = {
        'beta': 0.5,
        'delta': 0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'size': 10,
    }
    params_SI_beta_6 = {
        'fixed_prob': 0.1,
        'beta': 0.6,
        'delta': 0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        'size': 10,
    }

    params_probit_5 = {
        'sigma': 0.3,
        'theta': 1.5,
        'size': 10,
    }

    params_logit_5 = {
        'sigma': 0.3/1.7,
        'theta': 1.5,
        'size': 10,
    }

    params_linear_threshold = {
        'theta': 2,
        'size': 10,
    }

    params_sub_threshold_adoption = {
        'theta': 2,
        'size': 10,
        'fixed_prob': 0.05,
        'fixed_prob_high': 1.0,
        'zero_at_zer': True,
    }

    dynamics_4 = SIS(params_SI_beta_4)
    dynamics_5 = SIS(params_SI_beta_5)
    dynamics_6 = SIS(params_SI_beta_6)

    activation_probabilities_si_4 = dynamics_4.get_activation_probabilities(degree_range)

    activation_probabilities_si_5 = dynamics_5.get_activation_probabilities(degree_range)

    activation_probabilities_si_6 = dynamics_6.get_activation_probabilities(degree_range)

    activation_probabilities_si_4_int = dynamics_4.get_activation_probabilities(degree_range_int)

    activation_probabilities_si_5_int = dynamics_5.get_activation_probabilities(degree_range_int)

    activation_probabilities_si_6_int = dynamics_6.get_activation_probabilities(degree_range_int)

    plt.figure(1,(7, 6))

    plt.plot(degree_range, activation_probabilities_si_4, linewidth=2,
             label='$\\beta  =  0.4$')
    plt.scatter(degree_range_int, activation_probabilities_si_4_int)
    # plt.plot(degree_range, activation_probabilities_si_5, linewidth=2,
    #          label='$\\beta  =  0.5$')
    # plt.scatter(degree_range_int, activation_probabilities_si_5_int)
    plt.plot(degree_range, activation_probabilities_si_6, linewidth=2,
             label='$\\beta  =  0.6$')
    plt.scatter(degree_range_int, activation_probabilities_si_6_int)

    plt.ylabel('adoption probability', fontsize=20)
    plt.xlabel('number of neighboring adopters', fontsize=20)
    # plt.title('Simple Activation Functions', fontsize=20)
    plt.legend(loc='lower right', prop={'size': 20})
    # plt.show()
    plt.savefig("./figures/figure1A.pdf")

    plt.close(1)

    ########################
    ########################

    plt.figure(2,(7, 6))

    dynamics_probit = Probit(params_probit_5)
    dynamics_logit = Logit(params_logit_5)
    # dynamics_si_threshold = SIS_threshold(params_4)
    # dynamics_si_threshold_soft = SIS_threshold_soft(params_5)
    dynamics_linear = DeterministicLinear(params_linear_threshold)
    # dynamics_si_threshold_soft_fixed = SIS_threshold(params_6)
    dynamics_linear_sub_threshold = \
        DeterministicLinear(params_sub_threshold_adoption)

    activation_probabilities_probit = dynamics_probit.get_activation_probabilities(degree_range)

    activation_probabilities_logit = dynamics_logit.get_activation_probabilities(degree_range)

    # activation_probabilities_si_threshold =
    # dynamics_si_threshold.get_activation_probabilities(degree_range)
    #
    # activation_probabilities_si_threshold_soft =
    # dynamics_si_threshold_soft.get_activation_probabilities(degree_range)
    #
    # activation_probabilities_si_threshold_soft_fixed =
    # dynamics_si_threshold_soft_fixed.get_activation_probabilities(degree_range)

    activation_probabilities_linear = dynamics_linear.get_activation_probabilities(degree_range)

    activation_probabilities_sub_threshold = dynamics_linear_sub_threshold.get_activation_probabilities(degree_range)

    activation_probabilities_probit_int = dynamics_probit.get_activation_probabilities(degree_range_int)

    activation_probabilities_logit_int = dynamics_logit.get_activation_probabilities(degree_range_int)

    # activation_probabilities_si_threshold =
    # dynamics_si_threshold.get_activation_probabilities(degree_range)
    #
    # activation_probabilities_si_threshold_soft =
    # dynamics_si_threshold_soft.get_activation_probabilities(degree_range)
    #
    # activation_probabilities_si_threshold_soft_fixed =
    # dynamics_si_threshold_soft_fixed.get_activation_probabilities(degree_range)

    activation_probabilities_linear_int = \
        dynamics_linear.get_activation_probabilities(degree_range_int)

    activation_probabilities_sub_threshold_int = \
        dynamics_linear_sub_threshold.get_activation_probabilities(degree_range_int)

    print(degree_range)

    # plt.plot(degree_range, activation_probabilities_probit, linewidth=3, linestyle='--',
    #          label='probit $(\\sigma = 0.3)$')
    # plt.scatter(degree_range_int, activation_probabilities_probit_int)
    # plt.plot(degree_range, activation_probabilities_logit, linewidth=2,
    #          label='logit $(\\sigma = 0.18)$')
    # plt.scatter(degree_range_int, activation_probabilities_logit_int)
    plt.plot(degree_range, activation_probabilities_linear, 'r--', linewidth=1.5,
             label='deterministic\n 2-complex')
    plt.scatter(degree_range_int, activation_probabilities_linear_int)
    plt.plot(degree_range, activation_probabilities_sub_threshold, linewidth=4,
             label='noisy\n 2-complex')
    plt.scatter(degree_range_int, activation_probabilities_sub_threshold_int)
    # plt.plot(degree_range, activation_probabilities_si_threshold,
    # label='Threshold SI',alpha = 0.4, linewidth = 4)
    # plt.plot(degree_range, activation_probabilities_si_threshold_soft, ':', linewidth = 3,
    # label = 'Soft Thresholding of SI')
    # plt.plot(degree_range, activation_probabilities_si_threshold_soft_fixed,
    # label = 'Threshold SI (Fixed  Non-Zero)', alpha = 0.4 ,linewidth = 4)

    plt.ylabel('adoption probability', fontsize=20)
    plt.xlabel('number of neighboring adopters', fontsize=20)
    # plt.title('Complex Activation Functions', fontsize=20)
    plt.text(1.5, 0.1, '$q$', fontsize=30,
             fontweight='bold', fontdict=None, withdash=False)
    plt.legend(loc='lower right', prop={'size': 20})
    # plt.show()
    plt.savefig("./figures/figure1B.pdf")

    plt.close(2)

    plt.figure(3, (7.5, 6))

    plt.plot(degree_range, activation_probabilities_linear, 'r--', linewidth=1.5,
             label='deterministic\n 2-complex')
    plt.scatter(degree_range_int, activation_probabilities_linear_int)

    plt.plot(degree_range, activation_probabilities_logit, linewidth=2,
             label='logit \n$(\\sigma = 0.18)$')
    plt.scatter(degree_range_int, activation_probabilities_logit_int)
    plt.plot(degree_range, activation_probabilities_probit, linewidth=3, linestyle='--',
             label='probit \n$(\\sigma = 0.3)$')
    plt.scatter(degree_range_int, activation_probabilities_probit_int)

    plt.plot(degree_range, activation_probabilities_sub_threshold, linewidth=4,
             label='noisy\n 2-complex')
    plt.scatter(degree_range_int, activation_probabilities_sub_threshold_int)

    plt.ylabel('adoption probability', fontsize=20)
    plt.xlabel('number of neighboring adopters', fontsize=20)
    # plt.title('Complex Activation Functions', fontsize=20)
    plt.text(1.5, 0.1, '$q$', fontsize=30,
             fontweight='bold', fontdict=None, withdash=False)
    plt.legend(loc='lower right', prop={'size': 20})
    # plt.show()
    plt.savefig("./figures/figureS6A.pdf")

    plt.close(3)

    plt.figure(4,(7, 6))

    plt.plot(degree_range[0:180], activation_probabilities_sub_threshold[0:180], linewidth=4)
    plt.scatter(degree_range_int[0:3], activation_probabilities_sub_threshold_int[0:3])

    plt.ylabel('adoption probability', fontsize=20)
    plt.xlabel('number of neighboring adopters', fontsize=20)
    plt.ylim(ymax=1.3)
    plt.text(2.25, 1.1, '$\\rho$', fontsize=30,
             fontweight='bold', fontdict=None, withdash=False)
    plt.text(1.25, 0.1, '$q$', fontsize=30,
             fontweight='bold', fontdict=None, withdash=False)
    plt.text(2.05, 0.55, '$\\theta$', fontsize=30,
             fontweight='bold', fontdict=None, withdash=False)
    plt.title('A Noisy Complex Activation Function',fontsize=20)
    plt.xticks([0,1,2,3])
    # plt.show()
    plt.savefig("./figures/figureS9B.pdf")

    plt.close(4)
