
from models import *
from settings import *



degree_range=np.linspace(0,5,300)

if __name__ == '__main__':

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

    params_probit_logit_5 = {
        'sigma': 0.5,
        'theta': 1.5,
        'size': 10,
    }

    params_probit_logit_5 = {
        'sigma': 0.5,
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
        'fixed_prob': 0.15,
        'fixed_prob_high': 1.0,
        'zero_at_zer':True,
    }

    dynamics_4 = SIS(params_SI_beta_4)
    dynamics_5 = SIS(params_SI_beta_5)
    dynamics_6 = SIS(params_SI_beta_6)

    activation_probabilities_si_4 = dynamics_4.get_activation_probabilities(degree_range)

    activation_probabilities_si_5 = dynamics_5.get_activation_probabilities(degree_range)

    activation_probabilities_si_6 = dynamics_6.get_activation_probabilities(degree_range)

    plt.plot(degree_range, activation_probabilities_si_4, linewidth=2, label='$\\beta  =  0.4$')
    plt.plot(degree_range, activation_probabilities_si_5, linewidth=2, label='$\\beta  =  0.5$')
    plt.plot(degree_range, activation_probabilities_si_6, linewidth=2, label='$\\beta  =  0.6$')

    plt.ylabel('Adoption Probability')
    plt.xlabel('Number of Adopters in the Social Neighborhood')
    plt.title('Simple Activation Functions')
    plt.legend()
    plt.show()


    dynamics_probit = Probit(params_probit_logit_5)
    dynamics_logit = Logit(params_probit_logit_5)
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
    print(degree_range)

    plt.plot(degree_range, activation_probabilities_probit, linewidth=2, label='Probit $(\\sigma = 0.5)$')
    plt.plot(degree_range, activation_probabilities_logit, linewidth=2, label='Logit $(\\sigma = 0.5)$')
    plt.plot(degree_range, activation_probabilities_linear, 'r--', linewidth=1.5, label='Linear Threshold')
    plt.plot(degree_range, activation_probabilities_sub_threshold, linewidth=4, label='Sub-Threshold Adoptions')
    # plt.plot(degree_range, activation_probabilities_si_threshold,
    # label='Threshold SI',alpha = 0.4, linewidth = 4)
    # plt.plot(degree_range, activation_probabilities_si_threshold_soft, ':', linewidth = 3,
    # label = 'Soft Thresholding of SI')
    # plt.plot(degree_range, activation_probabilities_si_threshold_soft_fixed,
    # label = 'Threshold SI (Fixed  Non-Zero)', alpha = 0.4 ,linewidth = 4)

    plt.ylabel('Adoption Probability')
    plt.xlabel('Number of Adopters in the Social Neighborhood')
    plt.title('Complex Activation Functions')
    plt.legend()
    plt.show()