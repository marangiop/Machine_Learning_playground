import scipy.io as sio
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

#Section 1
#Loading data and plotting
amp=sio.loadmat('amp_data.mat')
ampdata=amp['amp_data'].squeeze()

plt.figure(figsize=(12,7))
plt.subplot(121)
plt.xlabel('Sequence')
plt.ylabel('Amplitude')
plt.title('Amplitude Sequence')
plt.plot(ampdata)
plt.subplot(122)
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of the Amplitudes')
plt.hist(ampdata,bins=100)
plt.show(block=False)

'''
In particular, two things seem to stand out for this data: 
In the histogram of amplitudes, we can see that most of the amplitude values recorded have zero frequency and the distribution of data points appears to be decaying exponentially as the distance from zero increases.
As seen in the sequence plot, over time time, deviations from the mean tend to occur in a correlated manner. It means that as the amplitude starts to deviate over time, the probability with which the amplitude deviates in the next time-step increases, creating deviation clusters over time.
'''


'''Discard some elements to ensure the 'amp_data' vector length is a multiple of 'lag'.
Wrap the amp data vector into a Cx'lag' matrix, where each row contains
'lag' amplitudes that were adjacent in the original sequence.'''

lag=21
reshaped_data=np.reshape(ampdata[len(ampdata)%lag:],(-1,lag))
X=reshaped_data[:,:-1]
y=reshaped_data[:,-1]

def gen_crossval_set(X,y,train_frac=.7,val_frac=.15,test_frac=.15):
    np.random.seed(1)
    indices=np.random.permutation(np.arange(len(X)))

    train_ind_start=0
    train_ind_end=int(len(X)*train_frac)

    val_ind_start=train_ind_end
    val_ind_end=val_ind_start+int(len(X)*val_frac)

    test_ind_start=val_ind_end
    test_ind_end=test_ind_start+int(len(X)*test_frac)

    train_is=indices[train_ind_start :train_ind_end ]
    val_is=indices[val_ind_start:val_ind_end]
    test_is=indices[test_ind_start:test_ind_end]

    return {
    "train":(X[train_is],y[train_is]),
    "val":(X[val_is],y[val_is]),
    "test":(X[test_is],y[test_is])
    }

datadict=gen_crossval_set(X,y)

#Split data intro train, val, test sets
X_shuf_train,y_shuf_train=datadict['train']
X_shuf_val,y_shuf_val=datadict['val']
X_shuf_test,y_shuf_test=datadict['test']

#Section 2
#Curve fitting (linear vs. quartic fit)
t=np.linspace(0,1,lag-1,endpoint=False)

x_example=X_shuf_train[3,None].T
y_example=y_shuf_train[3]

phi_linear=lambda x:np.vstack([np.ones(len(x)),x]).T
phi_quartic=lambda x:np.vstack([np.ones(len(x)),x,x**2,x**3,x**4]).T

fitted_linear_weights=np.linalg.lstsq(phi_linear(t),x_example,rcond=None)[0]
fitted_quartic_weights=np.linalg.lstsq(phi_quartic(t),x_example,rcond=None)[0]

t_with_test_point=np.append(t,1)
linear_fit=np.matmul(phi_linear(t_with_test_point),fitted_linear_weights)
quartic_fit=np.matmul(phi_quartic(t_with_test_point),fitted_quartic_weights)

#print(t_with_test_point)

xticks=[0.0,0.2,0.4,0.6,0.8,1.0]
plt.figure(figsize=(12,7))
plt.subplot(121)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Curve fitting audio amplitudes')
plt.xticks()
plt.plot(x_example,marker='o', label="Training Data")
plt.plot(linear_fit, label="Linear Fit")
plt.plot(quartic_fit, label="Quartic Fit")
plt.scatter(20,-0.05, color="r", label="Test Point")
plt.legend()
plt.show(block=False)


'''
The use of derivatives tells us that if we assume the true underlying function to be some nonlinear function, 
we can still model the smallest possible change using a linear function. In other words, we can assume that if we
we measure points that are closer together, their relationship will approach a linear function.

In order to estimate the gradient (which is the same as fitting a linear function) between the last point of a training set 
, we can take the last two know points into consideration. Considering more than two points has a a detrimental effect 
(i.e. it increases the loss function value) since these points don't originate from a linear function, and therefore (on average) 
do not additionally provide any useful information for estimating the gradient near the test point.

Using the same approach for estimating the test point using the quartic fit leads to failure. This is because it is not possible to
describe the complex curvature of a quartic function using such a small number of datapoints.

It would be naive to conclude that the quartic fit performs better than a linear function simply ecause it seems to closely follow
all the training data. While the quartic fit might seem to fit the training data well, in a number of cases
from the training set the quartic fit does not extrapolate well to the test point (i.e. 21st point). 
'''


#Calculating prediction for test data
##construct CxK design matrix, representing the C most recent time steps before the time we wish to predict, with K basis functions (polynomial)
##New prediction is a linear combination of previous amplitudes with vector v of weights
##Advantage is that we can compute vector v once and make next step predictions for N sequences of amplitudes without fitting N separate models


def phi(C,K):
    t=np.linspace(0,1,20,endpoint=False)[-C:]
    _,yv=np.meshgrid(np.arange(0,K),t)
    basispowers=np.arange(0,K)
    phi=np.power(yv,basispowers)
    return phi


def makevv(C,K):
    p=phi(C,K)
    v=np.matmul(np.matmul(p,np.linalg.inv(np.matmul(p.T,p))),np.ones((K,1)))
    return v

linearv=makevv(20,2)
quarticv=makevv(20,5)

lin=np.matmul(linearv.T,x_example)
qua=np.matmul(quarticv.T,x_example)

plt.figure(figsize=(12,7))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('CurveFittingAudioAmplitudes')
plt.plot(t,x_example,'-o',label='TrainingData')
plt.plot(1,y_example,'r.',label='TestPoint',markersize=10)
plt.plot(1,lin,'m.',label='LinearPredictionusingv',markersize=20)
plt.plot(1,qua,'c.',label='QuarticPredictionusingv',markersize=20)
plt.plot(t_with_test_point,linear_fit,'y',label='LinearFit')
plt.plot(t_with_test_point,quartic_fit,'g',label='QuarticFit')
plt.legend()
plt.show(block=False)

'''
This plot shows that using v derived in 3a to predict the next amplitude, achieves the same results as using a linear least squares fit.
We can also see that the quartic fit overshoots the test point. The quartic fit also undershoots in other cases (not shown here).
This suggests that the quartic model might be overtting the training data. 
On the other hand, the linear model is less prone to over- and undershooting.
'''

#Section 3
#Identify the best setting of K (number of basis function) and C (number of data points) on training set using grid search

cmax=20
kmax=10

errors=np.zeros((cmax,kmax))
for c in range(1,cmax+1):
    print('C:',c)
    for k in range(1,kmax+1):
        v=makevv(c,k)
        pred=np.matmul(v.T,X_shuf_train[:,-c:].T)
        sqerror=(pred-y_shuf_train)**2
        errors[c-1,k-1]=np.mean(sqerror)

ax=sns.heatmap(errors,linewidth=0.5,xticklabels=range(1,kmax+1), yticklabels=range(1,cmax+1),vmin=0,vmax=1e-3, cbar_kws={'label':'Average Mean Square Error'} )
plt.title('Grid Search for C and K Parameters on Training Set')
plt.ylabel('C')
plt.xlabel('K')
plt.legend()
plt.show(block=False)



print(np.min(errors))
minindices=np.unravel_index(np.argmin(errors),errors.shape)
print('C:',minindices[0]+1,'K:',minindices[1]+1)


#C = 2 and K = 2 achieves the lowest mean squared error on the training set.

bestv=makevv(2,2)
pred=np.matmul(bestv.T,X_shuf_train[:,-2:].T)
sqerror=(pred-y_shuf_train)**2
print('Train Squared Error',np.mean(sqerror))
pred=np.matmul(bestv.T,X_shuf_val[:,-2:].T)
sqerror=(pred-y_shuf_val)**2
print('Validation Squared Error',np.mean(sqerror))
pred=np.matmul(bestv.T,X_shuf_test[:,-2:].T)
sqerror=(pred-y_shuf_test)**2
print('Test Squared Error',np.mean(sqerror))

#Section 4
#Fitting linear predictors based on different numbers of datapoints

def get_linear_lstsq_for_diffc(x,y):
    maxc=20
    errors_c=np.zeros(maxc)
    for c in range(1,maxc+1):
        fitted_linear_weights=np.linalg.lstsq(X_shuf_train[:,-c:], y_shuf_train,rcond=None)[0]
        pred=np.matmul(fitted_linear_weights,x[:,-c:].T)
        sqerror=(pred-y)**2
        errors_c[c-1]=np.mean(sqerror)
    return errors_c

print(get_linear_lstsq_for_diffc(X_shuf_train,y_shuf_train))
print(get_linear_lstsq_for_diffc(X_shuf_val,y_shuf_val))

'''
On both the training and validation set, using the maximum context length (i.e. C = 20) for fitting a
linear least squares model resulted in the smallest mean square error.  

The shorter the context length C used to fit the linear least squares model, the higher the mean squared error was on
both the training and validation set. A reason as to why models fit using a longer context length
performed better than models gitt using shorter context lengths is that the longer the context
length, the more parameters the model has to tune in order to get a closer fit to the training data. 

It coul be expected that increasing the context length could led to overtting, but the
results on the validation set suggest that this did not occur. 

Looking at the chosen weights for fithe tted model using C = 20, it is noteworthy that the magnitude of the weights are smaller
the further away they are in time from the time-step we are predicting, which suggests that
the latter points are more important in making good predictions on the next time-step, which is also what one would expect.
'''


#Comparison of best linear predictor with best polynomial model on test set

c=20
fitted_linear_weights=np.linalg.lstsq(X_shuf_train[:,-c:],y_shuf_train,rcond=None)[0]

pred=np.matmul(fitted_linear_weights,X_shuf_test[:,-c:].T)
meansqerror=np.mean((pred-y_shuf_test)**2)
print('Test Mean Squared Error using Standard Linear lstsq',meansqerror)

best_v=makevv(2,2)
pred_v=np.matmul(bestv.T,X_shuf_test[:,-2:].T)
meansqerror_v=np.mean((pred_v-y_shuf_test)**2)
print('Test Mean Squared Error with C=2 and K=2',meansqerror_v)

'''

The standard linear least squares model (fit using context length C = 20) achieved a mean
squared error on the test set of 8.0267e-06, outperforming the best polynomial model (see section 3), which
achieved a mean squared error of 1.3760e-05 on the test set.
'''

#Histogram of the residuals on validation data for best model
c=20
fitted_linear_weights=np.linalg.lstsq(X_shuf_train[:,-c:],y_shuf_train,rcond=None)[0]
pred=np.matmul(fitted_linear_weights,X_shuf_val[:,-c:].T)

plt.figure(figsize=(12,7))
plt.xlim(-.10,.10)
plt.title('Validation set residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.hist(pred-y_shuf_val,bins=100)
plt.legend()
plt.show(block=False)

'''
In a similar fashion as the the distribution of the amplitudes (see first plot), the
distribution of the residuals is centered around 0, with most values very close to 0. 
This indicates that the mean of the residuals is apprximatly 0, which is expected when fitting linear model with least squares.
'''
