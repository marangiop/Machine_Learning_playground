#Asymptotic normality of Maximum Likelihood Estimate (MLE)


#The following code draws 5000 samples of size 150 from a Poisson(20) and each time time stores the MLE to the vector mle.
set.seed(400) # set a random seed for reproducibility of results
mle <- NULL
for (i in 1:5000){
       x <- rpois(n=150,20) # draw a sample of size 150 from Poisson(l=20)
        mle[i] <- mean(x) # This only works because the MLE for a poisson is the mean. MLE might not be equivalent to the mean for different for other distributions
        }
m <- mean(mle) #Calculate the sample mean of the MLE 
v <- var(mle) #Calculate the sample variance of the MLE 
c(m,v)


#The following code plots the histogram of the mle and the density of the normal distribution

hist(mle,xlab=expression(hat(lambda)),freq=FALSE)
abline(v=m,col=2, lty=2,lwd=2)
# Add the line of a normal curve with mean=20, sigma=0.134
z <- seq(18,22,length=1000)
lines(z,dnorm(z, 20, sqrt(0.134)),col=4)

#We can also verify the 95% CI we constructed using the simulated sample of the MLE. 
#The following code calculates the 97.5% and 2.5% quantile of the simulate sample and adds them on the previous plot
c(quantile(mle,0.025), quantile(mle,0.975))
abline(v=quantile(mle,0.975),col=2,lty=2,lwd=2)
abline(v=quantile(mle,0.025),col=2,lty=2,lwd=2)


