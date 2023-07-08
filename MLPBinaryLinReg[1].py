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

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""
    def accuracy(self, predicted, gold):
        return np.mean(predicted == gold)

def add_bias(X, bias):
    """X is a Nxm matrix: N datapoints, m features
    bias is a bias term, -1 or 1. Use 0 for no bias
    Return a Nx(m+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden
        self._losses =[]
        self._accuracies =[]
        self.epochs = 0
        self._val_losses =[]
        self._val_accuracies=[]
        
        def logistic(x):
            return 1/(1+np.exp(-x))
        self.activ = logistic
        
        def logistic_diff(y):
            return y * (1 - y)
        self.activ_diff = logistic_diff

    def calculate_loss_and_accuracy(self, X_train, t_train, outputs):
        loss = -np.mean(t_train * np.log(outputs) + (1 - t_train) * np.log(1 - outputs)) 
        accuracy = self.accuracy(self.predict(X_train), t_train)
        return loss, accuracy
        

    def fit_2(self, X_train, t_train, eta=0.001, epochs = 100):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Itilaize the wights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        
        for e in range(epochs):
            # One epoch
            hidden_outs, outputs = self.forward(X_train_bias)
            # The forward step
            out_deltas = (outputs - T_train)
            # The delta term on the output node
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The delta terms at the output of the jidden layer
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  
            # The deltas at the input to the hidden layer
            self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas 
            # Update the weights
            
    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = hidden_outs @ self.weights2
        return hidden_outs, outputs
    
    def predict(self, X):
        """Predict the class for the mebers of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)

    def predict_probability(self, X):
        """Predict the probability of a datapoint belonging to the positive class"""
        prob = self.predict(X)
        return 1 / (1 + np.exp(-prob))
    
    def calculate_loss_and_accuracy(self, X_train, t_train, outputs):
        loss = -np.mean(t_train * np.log(outputs) + (1 - t_train) * np.log(1 - outputs)) 
        accuracy = self.accuracy(self.predict(X_train), t_train)
        return loss, accuracy
    
    
    
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
    

#trener klassifikatoren med tidligere beste resultater, og henter ut 
# lister over loss og accuracy for hver epoch.  
"""epochs = 200
eta = 0.001

clf = MLPBinaryLinRegClass(bias=-1)
clf.fit(X_train, t2_train, epochs=epochs, eta=eta)"""



"""plt.figure(figsize=(8,6))
plot_decision_regions(X_train, t2_train, clf=clf)
plt.title('Training set with decision regions')
plt.show()"""

"""# plotter losses og accuracies som en funkjson av antall epochs. 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

ax[0].plot(losses)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('MSE')

ax[1].plot(accuracies)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
"""

"""plot_decision_regions(X_train, t2_train, clf)
plt.show()"""

"""learning_rates = [0.01, 0.001, 0.0001]
epochs = [50, 100, 200]

best_acc = 0
best_lr = None
best_epochs = None

for lr in learning_rates:
    for ep in epochs:
        model = MLPBinaryLinRegClass()
        model.fit_2(X_train, t2_train, eta=lr, epochs=ep)
        preds = model.predict(X_val)
        acc = model.accuracy(preds, t2_val)
        print(f"Learning rate: {lr}, Epochs: {ep}, Validation accuracy: {acc}")
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_epochs = ep

print(f"Best hyperparameters: Learning rate: {best_lr}, Epochs: {best_epochs}, Validation accuracy: {best_acc}")"""

"""clf_train = MLPBinaryLinRegClass()
clf_train.fit_2(X_train, t2_train, eta=0.0001, epochs=200)
preds = clf_train.predict(X_train)
acc = clf_train.accuracy(preds, t2_train) 
print("accuracy on training-set: ", acc)

clf_val = MLPBinaryLinRegClass()
clf_val.fit_2(X_val, t2_val, eta=0.0001, epochs=200)
preds = clf_val.predict(X_val)
acc = clf_val.accuracy(preds, t2_val) 
print("accuracy on validations-set: ", acc)

clf = MLPBinaryLinRegClass()
clf.fit_2(X_test, t2_test, eta=0.0001, epochs=200)
preds = clf.predict(X_test)
acc = clf.accuracy(preds, t2_test) 
print("accuracy on test-set: ", acc)"""

classifier = MLPBinaryLinRegClass()
classifier.fit_2(X_train, t2_train, eta=0.0001, epochs=200)

# Predict classes on the test set
y_pred = classifier.predict(X_test)

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

def fit(self, X_train, t_train, n_epochs,  X_val=None, t_val=None, tol=None, n_epochs_no_update=5):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Itilaize the wights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        
        for e in range(n_epochs):
            # One epoch
            hidden_outs, outputs = self.forward(X_train_bias)
            # The forward step
            out_deltas = (outputs - T_train)
            # The delta term on the output node
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The delta terms at the output of the jidden layer
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  
            # The deltas at the input to the hidden layer
            self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas 
            # Update the weights

            loss, acc = self.calculate_loss_and_accuracy(X_train, t_train, outputs)
            self._accuracies.append(acc)
            self._losses.append(loss)

            if X_val is not None and t_val is not None: 
                val_loss, val_acc = self.calculate_loss_and_accuracy(X_val, t_val)
                # append validation loss and accuracy to history
                self._val_losses.append(val_loss)
                self._val_accuracies.append(val_acc)

             # check if loss has not improved with more than tol after n_epochs_no_update
            if len(self._losses) > n_epochs_no_update:
                last_n_losses = self._losses[-n_epochs_no_update:]
                best_loss = min(self._losses[:-n_epochs_no_update])
                if all([loss > best_loss + tol for loss in last_n_losses]):
                    break  # early stopping

        self.epochs = e + 1  # record the number of epochs completed 
        #return self._accuracies