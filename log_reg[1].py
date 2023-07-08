import numpy as np
import matplotlib.pyplot as plt
import sklearn #for datasets

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt 


X, t_multi = make_blobs(n_samples=[400,400,400, 400, 400], 
                            centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                            cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5],
                            n_features=2, random_state=2022)

indices = np.arange(X.shape[0])
rng = np.random.RandomState(2022)
rng.shuffle(indices)
indices[:10]

X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:]
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]]

t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')



plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title('Multi-class set')

plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title('Binary set')

# sacling 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""
    def accuracy(self, predicted, gold):
        return np.mean(predicted == gold)
    
def accuracy(predicted, gold):
        return np.mean(predicted == gold)

def logistic(x):
    return 1/(1+np.exp(-x))  
    
    
def add_bias(X, bias):
    """X is a Nxm matrix: N datapoints, m features
    bias is a bias term, -1 or 1. Use 0 for no bias
    Return a Nx(m+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class NumpyLogRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
        #self._losses = [] # lagrer losses
        self._accuracies = [] # lagrer accuracies

        self._loss_history = []
        self.n_epochs = 0 

    
    def fit(self, X_train, t_train, eta = 0.1, tol=1e-4, n_epochs_no_update=5, n_epochs = 1000):
        """X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N, the targets values for the training data"""

        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)

        best_loss = np.inf
        epochs_no_update = 0 
        
        
        for e in range(n_epochs):

            y_pred = 1 / (1 + np.exp(-X_train @ weights))
            loss = -np.mean(t_train * np.log(y_pred) + (1 - t_train) * np.log(1 - y_pred))
            

            # check if the loss has improved
            if best_loss - loss > tol:
                best_loss = loss
                epochs_no_update = 0
            else:
                epochs_no_update += 1
                if epochs_no_update >= n_epochs_no_update:
                    break
            
            # your existing code for updating the model parameters
            
            # track the loss history
            self._loss_history.append(loss)
            self.n_epochs += 1

            # regner ut loss og accuracy, legger til i instansvarbler for hver iterasjon 
            acc = ((y_pred > 0.5) == t_train).mean()
            
            self._accuracies.append(acc)

            grad = X_train.T @ (y_pred - t_train) / N
            weights -= eta * grad
            
        #return self._loss_history, self._accuracies
        
    def predict(self, X, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        y_pred = 1 / (1 + np.exp(-X @ self.weights))
        return y_pred > threshold 
    
    def forward(self, X):
        return logistic(X @ self.weights)

    def predict_probability(self, x):
        """ Tar en matrise x med k datapunkter og returnerer 
        en vektor med k sannsyligheter for hvert datapunkt
        som tilhører den positive klassen """
        #x = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        z = add_bias(x, self.bias)
        return self.forward(z)
    
    def predict_proba(self, X):
        #self.weights.reshape(-1, 1)
        #X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        z = np.dot(X, self.weights) + self.bias
        y_prob = 1.0 / (1.0 + np.exp(-z))
        print(y_prob.shape)
        return y_prob
    
    """def predict_proba(self, X):
        #X is a Kxm matrix for some K>=1
        #predict the probability of class 1 for each point in X
        if self.bias:
            X = add_bias(X, self.bias)
        y_pred = 1 / (1 + np.exp(-X @ self.weights))
        proba = np.zeros((len(y_pred), 2))
        proba[:, 1] = y_pred
        proba[:, 0] = 1 - y_pred
        return proba"""


def predict(self, x, threshold=0.5):
    """X is a Kxm matrix for some K>=1
    predict the value for each point in X"""
    z = add_bias(x)
    return (self.forward(z) > threshold).astype('int') 



def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a make of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")

#multilayer, one vs. rest approach 

num_classes = 5 

classifiers = [NumpyLogRegClass() for i in range(num_classes)]

# Train each classifier
for i, clf in enumerate(classifiers):
    t2_train_binary = np.zeros_like(t2_train)
    t2_train_binary[t2_train == i] = 1
    clf.fit(X_train, t2_train_binary)

# Predict the class with the highest probability score for each test data point
y_pred = np.zeros_like(t2_test)
print(y_pred.shape)
for i, clf in enumerate(classifiers):
    y_pred_i = clf.predict_proba(X_test)[:, 1]
    print(y_pred_i.shape)
    y_pred[:, i] = y_pred_i

# The predicted class will be the one with the highest probability score
y_pred_class = np.argmax(y_pred, axis=1)

print(y_pred_class)


"""logreg = NumpyLogRegClass()
losses, accuracies = logreg.fit(X_train_scaled, t2_train, eta=0.1, epochs=100)
plt.plot(losses)
plt.title('Loss over epochs')
plt.show()
plt.plot(accuracies)
plt.title('Accuracy over epochs')
plt.show()

acc = accuracy(logreg.predict(X_val_scaled), t2_val)
print('Validation accuracy:', acc)"""

"""# Create and fit the classifier
clf = NumpyLogRegClass()
losses, accuracies = clf.fit(X_train, t2_train,eta = 0.1, tol=1e-4, n_epochs_no_update=5)
#print(losses)
#print(accuracies)
#print(epnu)

etas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] # different learning rates
tols = [0.00001, 0.0001, 0.001, 0.01, 0.1] # different values of tolerances

best_accuracy = 0.0
best_eta = None
best_tol = None

for eta in etas:
    for tol in tols:
        classifier = NumpyLogRegClass()
        classifier.fit(X_train_scaled, t2_train, eta=eta, tol=tol, n_epochs_no_update=5)
        y_pred = classifier.predict(X_val_scaled)
        accuracy_value = accuracy(t2_val, y_pred)
        
        if accuracy_value > best_accuracy:
            best_accuracy = accuracy_value
            best_eta = eta
            best_tol = tol
            
print("Best accuracy:", best_accuracy)
print("Best eta:", best_eta)
print("Best tol:", best_tol)"""

"""clf = NumpyLogRegClass()
train_loss, train_acc = clf.fit(X_train_scaled, t2_train, eta=0.1, tol=1e-05, n_epochs_no_update=5)

clf_2 = NumpyLogRegClass()
val_loss, val_acc = clf_2.fit(X_val_scaled, t2_val, eta=0.1, tol=1e-05, n_epochs_no_update=5)

# plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()"""

#from sklearn import predict_proba
#from sklearn.linear_model import LogisticRegression 

#multilayer, one vs. rest approach 

num_classes = 5 

classifiers = [NumpyLogRegClass() for i in range(num_classes)]

# Train each classifier
for i, clf in enumerate(classifiers):
    t2_train_binary = np.zeros_like(t2_train)
    t2_train_binary[t2_train == i] = 1
    clf.fit(X_train, t2_train_binary)

# Predict the class with the highest probability score for each test data point
y_pred = np.zeros_like(t2_test)
for i, clf in enumerate(classifiers):
    y_pred_i = clf.predict_probability(X_test)[:, 1]
    y_pred[:, i] = y_pred_i

# The predicted class will be the one with the highest probability score
y_pred_class = np.argmax(y_pred, axis=1)

print(y_pred_class)
