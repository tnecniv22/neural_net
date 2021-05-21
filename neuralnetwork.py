### Neural Network ###

class NeuralNetwork:
    def __init__(self, x, t, nh=4, no=1, alpha=0.01):
        
        #Input Shape
        self.dshape     = np.shape(x)
        self.ni         = self.dshape[0]    #number of input layers
        self.nh         = nh                #number of hidden layers
        self.no         = no                #number of output layers
        
        #Neural Network Inputs
        self.x          = x   #size: [ni x nData]
        self.t          = t   #size: [ni x nData]
        
        #Parameters
        self.alpha      = alpha
        
        #initialize weights and biases. 
        self.weights1   = np.random.rand(self.ni, nh)
        self.biases1    = np.random.rand(nh,1)
        self.weights2   = np.random.rand(nh,no)
        self.biases2    = np.random.rand(no,1)
        
        #Predicted Output
        self.output     = np.zeros(self.t.shape)
        self.trainloss  = 0
        self.testloss   = 0
        
    def splitdata(self, testratio=0.85):
        #Split Data into Training and Test Sets
        
        n_train = round(x.shape[1]*testratio)
        n_test = x.shape[1] - n_train

        indices = np.random.permutation(x.shape[1])
        train_idx, test_idx = indices[:n_train], indices[n_test:]
        self.x_train, self.x_test = self.x[:,train_idx], self.x[:,test_idx]
        self.t_train, self.t_test = self.t[:,train_idx], self.t[:,test_idx]
    
    def train(self):
        #For each Data: Forward and Backward Propogate.
        
        nData = self.x_train.shape[1]
        loss_arr = np.zeros([nData, 1])
        
        for i in range(nData):
            datain  = self.x_train[:,i]
            dataout = self.t_train[:,i]
            
            loss_arr[i] = self.feedforward(datain, dataout)
            self.backprop(datain, dataout)
        
        self.trainloss = np.mean(loss_arr)
    
    def testnet(self, indata):
        #Tests the Neural Network. 

        self.net1 = (np.dot(indata.T, self.weights1) + self.biases1.T).T
        self.out1 = sigmoid(self.net1)
        
        #Output Layer In and Out
        self.net2 = (np.dot(self.out1.T, self.weights2) + self.biases2.T).T
        self.output = linear(self.net2)
    
    def testcost(self):
        self.testloss = np.mean(l2loss(self.output, self.t_test))
    
    def feedforward(self, indata, outdata):
        #Forward Propogation.
        
        #Hidden Layer In and Out
        self.net1 = (np.dot(indata.T, self.weights1) + self.biases1.T).T
        self.out1 = sigmoid(self.net1)
        
        #Output Layer In and Out
        self.net2 = (np.dot(self.out1.T, self.weights2) + self.biases2.T).T
        self.out2 = linear(self.net2)
        
        #Calculate Cost
        loss = l2loss(self.out2, outdata)
        return loss
        
    def backprop(self, indata, outdata):
        #Backward Propogation.
        
        #output layer
        dEdO_2 = l2loss_d(self.out2,outdata)
        dOdI_2 = linear_d(self.net2)
        dIdW_2 = np.matlib.repmat(self.out1, 1, self.no)
        dEdI_2 = dEdO_2 * dOdI_2
        dEdW_2 = dEdI_2.T * dIdW_2
        
        self.weights2 = self.weights2 - dEdW_2 * self.alpha
        self.biases2 = self.biases2 - dEdI_2 * self.alpha
        
        #input layer
        dEdO_1 = np.sum([dEdI_2.T * self.weights2], axis = 1)
        dOdI_1 = sigmoid_d(self.net1)
        dIdW_1 = np.matlib.repmat(indata, 1, self.nh)
        dEdI_1 = dEdO_1 * dOdI_1
        dEdW_1 = dEdI_1.T * dIdW_1
        
        self.weights1 = self.weights1 - dEdW_1 * self.alpha
        self.biases1 = self.biases1 - dEdI_1 * self.alpha