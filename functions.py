### Activation and Loss Functions for NN ###

### Activation Functions

#Sigmoid
def sigmoid(x):
    return logistic.cdf(x)

def sigmoid_d(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

#Linear
def linear(x):
    return x

def linear_d(x):
    return 1

### Loss Functions

#L2 Loss
def l2loss(x,t):
    return np.sum(np.square(x-t))

def l2loss_d(x,t):
    return x-t;

#Cross Entropy
def cross_entropy(x, t):
    return -np.log(x[t==1])