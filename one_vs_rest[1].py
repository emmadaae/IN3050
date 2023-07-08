# gjør om den logitiske klasifikatoren til en multi-class klassifikatorer 

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

    
    def fit(self, X_train, t_train, eta = 0.1, epochs=100, tol=1e-4, n_epochs_no_update=5, n_epochs = 1000):
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
        #X is a Kxm matrix for some K>=1
        #predict the value for each point in X
        if self.bias:
            X = add_bias(X, self.bias)
        y_pred = 1 / (1 + np.exp(-X @ self.weights))
        return y_pred > threshold

    """def predict(self, X):
        # Calculate predicted values using sigmoid function
        y_pred = logistic(X @ self.weights)
        # Classify as 1 or 0 based on whether predicted value is above or below 0.5 threshold
        return (y_pred >= 0.5).astype(int)"""
    
    def forward(self, X):
        return logistic(X @ self.weights)

    def predict_probability(self, x):
        #Tar en matrise x med k datapunkter og returnerer 
        #en vektor med k sannsyligheter for hvert datapunkt
        #som tilhører den positive klassen 
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


"""def predict(self, x, threshold=0.5):
    #X is a Kxm matrix for some K>=1
    #predict the value for each point in X
    z = add_bias(x)
    return (self.forward(z) > threshold).astype('int') """



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




class LogRegMulti(): 
    def __init__(self):
        self._classifiers = []

    def fit(self, X_train, t_train, eta=0.1, epochs=100, X_val=None, t_val = None, tol=1e-4, n_epochs_no_update=5):
        K = np.max(t_train) +1

        for k in range(K): 
            t_train_k = (t_train==k).astype(int)
            clf = NumpyLogRegClass()
            clf.fit(X_train, t_train_k, eta=eta, epochs=epochs, tol=tol, n_epochs_no_update=n_epochs_no_update)
            self._classifiers.append(clf)
    
    def predict(self, X):
        probs = np.array([clf.predict_probability(X) for clf in self._classifiers])
        # Choose class with highest probability for each data point
        return np.argmax(probs, axis=0)

    def predict_probability(self, X): 
        probabilities = np.array([clf.predict_probability(X) for clf in self._classifiers])
        return probabilities 
"""
etas3 = np.linspace(0.001, 0.01, num=100) 
etas2 = np.linspace(0.01, 0.1, num=200) 
etas1 = np.linspace(0.1, 1, num=200) 

etas_combined = np.concatenate(( etas3, etas2, etas1))

epochs = np.linspace(195, 205, num=10, dtype=int)
results = []
for eta in etas_combined: 
    for epoch in epochs: 
        cl = NumpyLogRegClass() 
        cl.fit(X_train_scaled, t2_train, eta=eta, epochs=epoch)
        acc = accuracy(cl.predict(X_train_scaled), t2_train)
        results.append([acc, eta, epoch])
    
results.sort(reverse=True)
best_acc, best_eta, best_epoch = results[0]

print("\n best parameters:")
print("eta:", best_eta, "epochs: ", best_epoch, "accuracy: ", best_acc)
#eta: 1.0 epochs:  205 accuracy:  0.763 """

"""
#kjører for multi-logreg med beste hyperparametere 
best_eta = 1.0
best_epoch = 205

clf = LogRegMulti()
clf.fit(X_train_scaled, t2_train, eta=best_eta, epochs=best_epoch)

y_pred = clf.predict(X_val)
acc = accuracy(y_pred, t_multi_val)
print(f"Accuracy on validation set: {acc:.2f}")
print(f"Accuracy: {accuracy(clf.predict(X_train), t_multi_train):.3f}")
plot_decision_regions(X_train, t_multi_train, clf)
#plt.show()"""

clf_train = NumpyLogRegClass()
clf_train.fit(X_train, t2_train, eta=1.0, epochs=205)
preds = clf_train.predict(X_train)
acc = clf_train.accuracy(preds, t2_train) 
print("accuracy on training-set: ", acc)

clf_val = NumpyLogRegClass()
clf_val.fit(X_val, t2_val, eta=1.0, epochs=205)
preds = clf_val.predict(X_val)
acc = clf_val.accuracy(preds, t2_val) 
print("accuracy on validations-set: ", acc)

clf = NumpyLogRegClass()
clf.fit(X_test, t2_test, eta=1.0, epochs=205)
preds = clf.predict(X_test)
acc = clf.accuracy(preds, t2_test) 
print("accuracy on test-set: ", acc)

classifier = NumpyLogRegClass()
classifier.fit(X_train_scaled, t2_train, eta=1.0, epochs=205)

# Predict classes on the test set
y_pred = classifier.predict(X_test_scaled)

# Get the true classes for the first class
y_true = t2_test == 0

# Calculate true positives, false positives, and false negatives for the first class
tp = np.sum((y_pred == 0) & (y_true == 0))
fp = np.sum((y_pred == 0) & (y_true == 1))
fn = np.sum((y_pred == 1) & (y_true == 0))

# Calculate precision and recall for the first class
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Precision for the first class:", precision)
print("Recall for the first class:", recall)
