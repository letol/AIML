import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


# color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_training_graph(xx, yy, model, X, y):
    # Plot the decision boundaries
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# %% Initialization
print("\n\t- Initialization")

all_features, target = load_wine(return_X_y=True)
features = all_features[:, :2]
features_train_val, features_test, target_train_val, target_test = train_test_split(features, target, test_size=.3)
features_train, features_val, target_train, target_val = train_test_split(features_train_val, target_train_val,
                                                                          test_size=(2 / 7))

# initializing the random seed
# np.random.seed(0)

# step size in the mesh
h = .02

# prepare the mesh
x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

x, y = np.meshgrid(np.arange(x_min, x_max, h),
                   np.arange(y_min, y_max, h))

# %% K-Nearest Neighbor
print("\n\t- K-Nearest Neighbor")

K = [1, 3, 5, 7]

# initialize accuracies list and best variables
accuracy = []
best_score = 0
best_K = 1

for i, k in enumerate(K):
    # Apply k-NN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features_train, target_train)

    # Plot the data and tha decision boundaries
    plt.figure()
    plot_training_graph(x, y, model, features_train, target_train)
    plt.title("K-Nearest Neighbor (k = %i)" % k)
    plt.show()

    # Evaluate on the validation set
    score = model.score(features_val, target_val)
    accuracy.append(score)

    if score > best_score:
        best_score = score
        best_K = k

# Plot accuracy variation
plt.figure()
plt.ylim(0.5, 1)
plt.ylabel("accuracy")
plt.xlabel("K")
plt.plot(K, accuracy, '-', K, accuracy, 'o')
plt.title("k-NN accuracy on validation set")
plt.show()

# Apply K-NN with best K on train+val set
clf1 = KNeighborsClassifier(n_neighbors=best_K)
clf1.fit(features_train_val, target_train_val)

# Evaluate the model on test set
accuracy_test = clf1.score(features_test, target_test)
print("Best K is %i and corresponding accuracy on test set is %f%%" % (best_K, accuracy_test * 100))

# %% Linear SVM
print("\n\t- Linear SVM")

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# initialize accuracies list and best variables
accuracy.clear()
best_score = 0
best_C = 1

for i, c in enumerate(C):
    # Apply Linear SVM
    model = LinearSVC(C=c)
    model.fit(features_train, target_train)

    # Plot the data and tha decision boundaries
    plt.figure()
    plot_training_graph(x, y, model, features_train, target_train)
    plt.title("Linear SVM (C = %f)" % c)
    plt.show()

    # Evaluate on the validation set
    score = model.score(features_val, target_val)
    accuracy.append(score)

    if score > best_score:
        best_score = score
        best_C = c

# Plot accuracy variation
plt.figure()
plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("Linear SVM accuracy on validation set")
plt.show()

# Apply Linear SVC with best C on train+val set
clf2 = LinearSVC(C=best_C)
clf2.fit(features_train_val, target_train_val)

# Evaluate the model on test set
accuracy_test = clf2.score(features_test, target_test)
print("Best C is %f and corresponding accuracy on test set is %f%%" % (best_C, accuracy_test * 100))

# %% SVM with RBF kernel
print("\n\t- SVM with RBF kernel")

# initialize accuracies list and best variables
accuracy.clear()
best_score = 0
best_C = 1

for i, c in enumerate(C):
    # Apply SVM
    model = SVC(C=c, gamma='auto', kernel='rbf')
    model.fit(features_train, target_train)

    # Plot the data and tha decision boundaries
    plt.figure()
    plot_training_graph(x, y, model, features_train, target_train)
    plt.title("SVM with RBF kernel (C = %f)" % c)
    plt.show()

    # Evaluate on the validation set
    score = model.score(features_val, target_val)
    accuracy.append(score)

    if score > best_score:
        best_score = score
        best_C = c

# Plot accuracy variation
plt.figure()
plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("SVM with RBF kernel accuracy on validation set")
plt.show()

# Apply SVC with best C on train+val set
clf3 = SVC(C=best_C, gamma='auto', kernel='rbf')
clf3.fit(features_train_val, target_train_val)

# Evaluate the model on test set
accuracy_test = clf3.score(features_test, target_test)
print("Best C is %f and corresponding accuracy on test set is %f%%" % (best_C, accuracy_test * 100))

# %% SVM with RBF kernel and grid search
print("\n\t- SVM with RBF kernel and grid search")

gamma_min = -7
gamma_max = -1
n_g = 7

C_min = -3
C_max = 3
n_C = 7

# Prepare the parameters grid
param_grid = {'C': np.logspace(C_min, C_max, n_C), 'gamma': np.logspace(gamma_min, gamma_max, n_g), 'kernel': ['rbf']}

best_score = 0
best_params = {'C': C_min, 'gamma': gamma_min, 'kernel': 'rbf'}

# Search for best parameters
for params in ParameterGrid(param_grid):
    model = SVC(**params)
    model.fit(features_train, target_train)
    score = model.score(features_val, target_val)
    if score > best_score:
        best_score = score
        best_params = params

# Apply SVC with pest C and gamma on train+val set
clf4 = SVC(**best_params)
clf4.fit(features_train_val, target_train_val)

# Plot the data and tha decision boundaries
plt.figure()
plot_training_graph(x, y, clf4, features_train, target_train)
plt.title("SVM with RBF kernel (C = %f, gamma = %f)" % (best_params['C'], best_params['gamma']))
plt.show()

# Evaluate the model on test set
accuracy_test = clf4.score(features_test, target_test)
print("Best params are C=%f and gamma=%f and corresponding accuracy on test set is %f%%"
      % (best_params['C'], best_params['gamma'], accuracy_test * 100))

# %% K-Fold
print("\n\t- K-Fold")

# Prepare folds
kf = KFold(n_splits=5)

# Perform grid search with cross validation
clf5 = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=kf.split(features_train_val), iid=False)
clf5.fit(features_train_val, target_train_val)

# Obtain best parameters
best_params = clf5.best_params_

# Plot the data and tha decision boundaries
plt.figure()
plot_training_graph(x, y, clf5, features_train, target_train)
plt.title("SVM with RBF and 5-Fold CV (C = %f, gamma = %f)" % (best_params['C'], best_params['gamma']))
plt.show()

# Evaluate the best model on test set
accuracy_test = clf5.score(features_test, target_test)
print("Best params are C=%f and gamma=%f and corresponding accuracy on test set is %f%%"
      % (best_params['C'], best_params['gamma'], accuracy_test * 100))
