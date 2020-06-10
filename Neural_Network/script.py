import numpy as np
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
# Prediction of Relative location of Computerised Tomography (CT) slice based on human body's axial axis Data Set
# From UCI Machine Learning Repository 
# https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis
# First 73 patients were put in the train arrays, the next 12 in the val arrays, and the final 12 in the test arrays

### PART 1 ###
#Load and extract raw data
rawdata=loadmat('ct_data.mat')
Xtrainraw=rawdata['X_train']
ytrainraw=rawdata['y_train']
Xvalraw=rawdata['X_val']
yvalraw=rawdata['y_val']
Xtestraw=rawdata['X_test']
ytestraw=rawdata['y_test']

#Squeeze y's to avoid memory issues later when calculating gradients
ytrain,yval,ytest=ytrainraw.squeeze(),yvalraw.squeeze(),ytestraw.squeeze()

#Calculate mean of y train --> zero
print('Mean of y train: {}'.format(ytrain.mean()))


# Calculate mean and SE of y val 
yvalmu=yval.mean()
yvalse=yval.std()/np.sqrt(len(yval))
print("Mean and standard error of yval:{},{}".format(yvalmu,yvalse))

# Calculate mean and SE of y val truncated on the first 5785 values
ytraintruncmu=ytrain[:5785].mean()
ytraintruncse=ytrain[:5785].std()/np.sqrt(len(ytrain[:5785]))
print("Mean and standard error of the first 5785 values of ytrain:{},{}".format(ytraintruncmu,ytraintruncse))

sns.distplot(yval, kde=False, bins=500)
#plt.show(block=False)
plt.show()

'''
y train's mean and SE for the first 5785 points are -0.442 and 0.0119, respectively.
The usual practice is to plot SE bars on either side of a sample's mean. This implies data symmetry.
However, y val is not symmetrically distributed around its mean. This is illustrated in the figure:
most of the values are centered around -0.8 (instead of the mean of the values). Nonetheless, the 
function does not taper off symmetrically to both sides (like it is seen in a Gaussian distribution),
and appears to be skewed to the left (most mass is larger than the mean).
Simply shortening the lower error bar will still give a misleading impression of the
data: this is because the error bar assumes a monotonic taper of the data, as we get farther away from
the mean. 

In the case of our dataset, the taper of the data seems to be more step-wise (e.g. data frequency appears to
drop off between 0 and 1, and stays constant afterwards). If we were to draw error bars around
the mean, our data might have some regime change within that error bar, while the error bar
predicts a monotonic taper.
'''

#findcolumnswithonlyoneuniquevalue
redundantcols=[]
for column in range(Xtrainraw.shape[1]):

    xtrcols=Xtrainraw[:,column]
    #if there is only one unique value in this column, select it for removal
    if len(np.unique(xtrcols))==1:
        redundantcols.append(column)

#find columns that are duplicates of other columns
cols=np.arange(Xtrainraw.shape[1])

#get all the indices of the unique columns
_,invertableind=np.unique(Xtrainraw,axis=1,return_index=True)

#take the inverse of the unique columns
duplicatecols=cols[~np.isin(cols,invertableind)]
cols_to_be_removed=np.unique(list(duplicatecols)+redundantcols)
print(cols_to_be_removed)


#Function to remove a list of columns from matrix, given their column indices, using a boolean mask
def removecolfrommatrix(cols,X):

    totalcollist=np.ones(X.shape[1])==1
    totalcollist[cols]=False

    return X[:,totalcollist]

Xtrain=removecolfrommatrix(cols_to_be_removed,Xtrainraw)
Xval=removecolfrommatrix(cols_to_be_removed,Xvalraw)
Xtest=removecolfrommatrix(cols_to_be_removed,Xtestraw)

#### PART 2 ###
#Linear Regression
##Fitting weights of LR model using both a regularization constant alpha and a bias b, without regularizing the bias; and least square.
# 
def fitlinreg(X,yy,alpha):
    #do some shape manipulation if the incoming shape is incorrect
    if len(np.shape(yy))<2:
        yy=yy[:,None]

    #add the 1'scolumn ,which will fit a bias
    biascol=np.ones((X.shape[0],1))
    Xbiased=np.concatenate((biascol,X),axis=1)

    #append sqrt(sigma)*identity to our target data to regalurize
    yyreg=np.concatenate((yy,np.zeros((X.shape[1]+1,1))),axis=0)
    alphamat=np.sqrt(alpha)*np.eye(X.shape[1]+1)

    #set regularization matrix[0,0] so we don't regularize the bias
    alphamat[0,0]=0.

    #put everything together and fit
    Xbiasedreg=np.concatenate((Xbiased,alphamat),axis=0)
    return np.linalg.lstsq(Xbiasedreg,np.squeeze(yyreg),rcond=0)


ws,_,_,_=fitlinreg(Xtrain,ytrain,10.)
ypredtrain=np.dot(ws[1:],Xtrain.T)+ws[0]
ypredval=np.dot(ws[1:],Xval.T)+ws[0]

print("Ridge regression baseline train RMSE: {}".format(sqrt(mean_squared_error(ypredtrain,ytrain))))
print("Ridge regression baseline val RMSE: {}".format(sqrt(mean_squared_error(ypredval,yval))))

#Weights of LR model can also be fitted using gradient descent.
#A property of LR is that both least squares and gradient descent give the unique optimal weights.


### PART 3 INVENTED CLASSIFICATION TASK ###

'''

def fit_logreg_gradopt(X,yy,alpha):
    '
    Find weights and bias by using a gradient-based optimizer 
    to improve the regularized negative log likelihood

    Inputs:
    X N,D design matrix of input features
    yy N, real-valuedtargets
    alpha scalar regularization constant

    Outputs:
    ww D, fitted weights
    bb    scalar fitted bias
    '
    D=X.shape[1]
    args=(X,yy,alpha)
    init=(np.zeros(D),np.array(0))
    ww,bb=ct.minimizelist(ct.logregcost,init,args)
    return ww,bb

def nll(X,yy,ww,bb,alpha):
    yy=2*(yy==1)-1

    aa=yy*(np.dot(X,ww)+bb)
    sigma=1/(1+np.exp(-aa))
    E=-np.sum(np.log(sigma))+alpha-np.dot(ww,ww)
    return E

def sigm(x):
    return 1/(1+np.exp(-x))


def generatesigmdata(Xtrain,ytrain):
    results=[]
    K=10 #number of thresholded classification problems to fit
    mx=np.max(ytrain); mn=np.min(ytrain); hh=(mx-mn)/(K+1)
    thresholds=np.linspace(mn+hh,mx-hh,num=K,endpoint=True)
    #preallocate for speed
    ws=np.empty((K,Xtrain.shape[1]))
    bs=np.empty((K,))
    for kk in range(K):

        labels=ytrain>thresholds[kk]

        ww,bb=fit_logreg_gradopt(Xtrain,labels,10.)

        ws[kk]=ww
        bs[kk]=bb

        return ws,bs


trainlogws,trainlogbs=generatesigmdata(Xtrain,ytrain)

vallogistprobs=sigm(Xval.dot(trainws.T)+trainbs)
trainlogistprobs=sigm(Xtrain.dot(trainws.T)+trainbs)

wslogisttrain=fitlinreg(trainlogistprobs,ytrain,10.)[0]

print(rmse(wslogisttrain[1:].dot(trainlogistprobs.T)+wslogisttrain[0],ytrain))
print(rmse(wslogisttrain[1:].dot(vallogistprobs.T)+wslogisttrain[0],yval))

'''