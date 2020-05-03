#This tutorial shows how we can use Bayes factors and marginal likelihood in order to perform model selection.
#Here we are working with a beta=binomial model for coin-flipping. 
#Marginal likelihood can be computed both analytically and with Sequential Monte Carlo



import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import betaln
from scipy.stats import beta

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))


###Analytically

#Our data consist on 100 “flips of a coin” and the same number of observed “heads” and “tails”. 
# We will compare two models: one with a uniform prior and one with a more concentrated prior around theta=0.5
def beta_binom(prior, y):
    """
    Compute the marginal likelihood, analytically, for a beta-binomial model.

    prior : tuple
        tuple of alpha and beta parameter for the prior (beta distribution)
    y : array
        array with "1" and "0" corresponding to the success and fails respectively
    """
    alpha, beta = prior
    h = np.sum(y)
    n = len(y)
    p_y = np.exp(betaln(alpha + h, beta+n-h) - betaln(alpha, beta))
    return p_y

y = np.repeat([1, 0], [50, 50])  # 50 "heads" and 50 "tails"
priors = ((1, 1), (30, 30))

for a, b in priors:
    distri = beta(a, b)
    x = np.linspace(0, 1, 100)
    x_pdf = distri.pdf(x)
    plt.plot (x, x_pdf, label=r'$\alpha$ = {:d}, $\beta$ = {:d}'.format(a, b))
    plt.yticks([])
    plt.xlabel('$\\theta$')
    plt.legend()

#Calculation of Bayes Factor
BF = (beta_binom(priors[1], y) / beta_binom(priors[0], y))
print(round(BF))

#We see that the model with the more concentrated prior Beta(30,30) has ≈5 times more support than the model with the more extended prior Beta(1,1).
#Besides the exact numerical value this should not be surprising since the prior for the most favoured model is concentrated around thetha =0.5 and the data y has equal number of head and tails, which is consintent with a value of θ around 0.5.

###Sequential Monte Carlo

n_chains = 1000

models = []
traces = []
for alpha, beta in priors:
    with pm.Model() as model:
        a = pm.Beta('a', alpha, beta)
        yl = pm.Bernoulli('yl', a, observed=y)
        trace = pm.sample_smc(1000,
                          random_seed=42)
        models.append(model)
        traces.append(trace)

BF_smc = models[1].marginal_likelihood / models[0].marginal_likelihood
print(round(BF_smc))

#SMC gives essentially the same answer as the analytical calculation.
#The advantage of using SMC is that we can use it to compute the marginal likelihood for a wider range of models, given that we do not need an analytical expression for the marginal likelihood. 


#Posterior from two models
print(pm.summary(traces[0], var_names='a').round(2))
print(pm.summary(traces[1], var_names='a').round(2))


#The results are very similar. we have the same mean value for theta, and a slightly wider posterior for model_0, as expected since this model has a wider prior.
# 



_ , ax = plt.subplots(figsize=(9, 6))
ppc_0 = pm.sample_posterior_predictive(traces[0], 100, models[0], size=(len(y), 20))
ppc_1 = pm.sample_posterior_predictive(traces[1], 100, models[1], size=(len(y), 20))
for m_0, m_1 in zip(ppc_0['yl'].T, ppc_1['yl'].T):
    pm.kdeplot(np.mean(m_0, 0), ax=ax, plot_kwargs={'color':'C0'})
    pm.kdeplot(np.mean(m_1, 0), ax=ax, plot_kwargs={'color':'C1'})
ax.plot([], label='model_0')
ax.plot([], label='model_1')
ax.legend(fontsize=14)
ax.set_xlabel(u'θ', fontsize=14)
ax.set_yticks([])
plt.show()

#In this example the observed data y is more consistent with model_1 (because the prior is concentrated around the correct value of heta) than model_0 (which assigns equal probability to every possible value of heta)/
#This difference is captured by the Bayes factors. We could say Bayes factors are measuring which model, as a whole, is better, including details of the prior that may be irrelevant for parameter inference. 
# In fact in this example we can also see that it is possible to have two different models, with different Bayes factors, but nevertheless get very similar predictions. The reason is that the data is informative enough to reduce the effect of the prior up to the point of inducing a very similar posterior.