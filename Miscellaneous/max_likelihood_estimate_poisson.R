####Maximisation of log-likelihood for Poisson ditribution

set.seed(600) # set a random seed for reproducibility of results
data <- rpois(n=150,20)# draw a sample of size 150 from Poisson(l=20)

poisson_loglikelihood <- function(lambda,x)
{
    n <-length(x) # n is the sample size
    llik <- -n*lambda+sum(x*log(lambda))-sum(log(factorial(x)))#Log-likelihood of Poisson
    return(llik)
}

maximisation <- optimize(poisson_loglikelihood,
                         interval=c(1,50),x=data,maximum=TRUE) 

maximisation # This will return the parameter value that maximises the functio and the maximum value of the function
