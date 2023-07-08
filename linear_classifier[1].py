
import numpy as np
import matplotlib.pyplot as plt
import sklearn #for datasets

from sklearn.datasets import make_blobs

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

def add_bias(X, bias):
    """X is a Nxm matrix: N datapoints, m features
    bias is a bias term, -1 or 1. Use 0 for no bias
    Return a Nx(m+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""
    """def accuracy(self, predicted, gold):
        return np.mean(predicted == gold)"""

class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
        self._losses = [] # lagrer losses
        self._accuracies = [] # lagrer accuracies
    
    def fit(self, X_train, t_train, eta = 0.1, epochs=10):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        

        for e in range(epochs):
            # regner ut loss og accuracy, legger til i instansvarbler for 
            # hver iterasjon 
            y_pred = X_train @ weights 
            loss = ((y_pred-t_train)**2).mean()
            self._losses.append(loss)

            acc = ((y_pred > 0.5) == t_train).mean()
            self._accuracies.append(acc)

            weights -= eta / N *  X_train.T @ (X_train @ weights - t_train)
        return self._losses, self._accuracies       
    
    def predict(self, X, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold

def accuracy(predicted, gold):
    return np.mean(predicted == gold)

cl = NumpyLinRegClass()
cl.fit(X_train, t2_train)
acc = accuracy(cl.predict(X_val), t2_val)
print(acc)

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


"""plot_decision_regions(X_train, t2_train, cl)
plt.show()"""

"""etas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] # different learning rates
epochs = [1, 10, 50, 100] # different number of epochs

best_accuracy = 0.0
best_eta = None
best_epochs = None

for eta in etas:
    for epoch in epochs:
        classifier = NumpyLinRegClass()
        classifier.fit(X_train, t2_train, eta=eta, epochs=epoch)
        y_pred = classifier.predict(X_val)
        accuracy_value = accuracy(t2_val, y_pred)
        
        if accuracy_value > best_accuracy:
            best_accuracy = accuracy_value
            best_eta = eta
            best_epochs = epoch
            
print("Best accuracy:", best_accuracy)
print("Best eta:", best_eta)
print("Best epochs:", best_epochs)"""


"""classifier = NumpyLinRegClass()
classifier.fit(X_train, t2_train, eta=0.00001, epochs=1)
plot_decision_regions(X_train, t2_train, classifier)
plt.show()"""

"""#trener klassifikatoren med tidligere beste resultater, og henter ut 
# lister over loss og accuracy for hver epoch.  
epochs = 200
eta = 0.01

clf = NumpyLinRegClass(bias=-1)
losses, accuracies = clf.fit(X_train_scaled, t2_train, epochs=epochs, eta=eta)
#print(losses)

# plotter losses og accuracies som en funkjson av antall epochs. 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

ax[0].plot(losses)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('MSE')

ax[1].plot(accuracies)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')

plt.show()"""

#tester med ny scaling 

"""etas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] # different learning rates
epochs = [1, 10, 50, 100, 200, 300, 400, 500, 750, 1000] # different number of epochs

best_accuracy = 0.0
best_eta = None
best_epochs = None

for eta in etas:
    for epoch in epochs:
        classifier = NumpyLinRegClass()
        classifier.fit(X_train_scaled, t2_train, eta=eta, epochs=epoch)
        y_pred = classifier.predict(X_val_scaled)
        accuracy_value = accuracy(t2_val, y_pred)
        
        if accuracy_value > best_accuracy:
            best_accuracy = accuracy_value
            best_eta = eta
            best_epochs = epoch
            
print("Best accuracy:", best_accuracy)
print("Best eta:", best_eta)
print("Best epochs:", best_epochs)"""

"""clf_train = NumpyLinRegClass()
clf_train.fit(X_train, t2_train, eta=0.0001, epochs=200)
preds = clf_train.predict(X_train)
acc = accuracy(preds, t2_train) 
print("accuracy on training-set: ", acc)

clf_val = NumpyLinRegClass()
clf_val.fit(X_val, t2_val, eta=0.0001, epochs=200)
preds = clf_val.predict(X_val)
acc = accuracy(preds, t2_val) 
print("accuracy on validations-set: ", acc)

clf = NumpyLinRegClass()
clf.fit(X_test, t2_test, eta=0.0001, epochs=200)
preds = clf.predict(X_test)
acc = accuracy(preds, t2_test) 
print("accuracy on test-set: ", acc)

from sklearn.metrics import precision_score, recall_score

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate precision and recall for class 1
precision = precision_score(t2_test, y_pred, pos_label=0)
recall = recall_score(t2_test, y_pred, pos_label=0)

print("Precision for class 1: {:.2f}".format(precision))
print("Recall for class 1: {:.2f}".format(recall))"""

classifier = NumpyLinRegClass()
losses, accuracies = classifier.fit(X_train_scaled, t2_train, eta=0.01, epochs=200)

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



