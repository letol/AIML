import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

# %% Initialization
print("\n\t- Initialization")

all_features, target = load_wine(return_X_y=True)
features = all_features[:, :2]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=.3)
features_train, features_val, target_train, target_val = train_test_split(features_train, target_train,
                                                                          test_size=(2 / 7))

# initializing the random seed for better debugging
# np.random.seed(0)

# step size in the mesh
h = .02

# color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# prepare the mesh
x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

x, y = np.meshgrid(np.arange(x_min, x_max, h),
                   np.arange(y_min, y_max, h))

# %% K-Nearest Neighbor
print("\n\t- K-Nearest Neighbor")

K = [1, 3, 5, 7]

# initialize classifiers and accuracies lists
clf1 = []
accuracy = []

for i, k in enumerate(K):
    # Apply k-NN
    clf1.append(KNeighborsClassifier(n_neighbors=k))
    clf1[i].fit(features_train, target_train)

    # Plot the decision boundaries
    Z = clf1[i].predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("K-Nearest Neighbor (k = %i)" % k)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(clf1[i].score(features_val, target_val))

plt.ylim(0.5, 1)
plt.ylabel("accuracy")
plt.xlabel("K")
plt.plot(K, accuracy, '-', K, accuracy, 'o')
plt.title("k-NN accuracy on validation set")
plt.show()

best_K_index = np.argmax(accuracy)
accuracy_test = clf1[best_K_index].score(features_test, target_test)
print("Best K is %i and corresponding accuracy on test set is %f%%" % (K[best_K_index], accuracy_test * 100))

# %% Linear SVM
print("\n\t- Linear SVM")

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# initialize classifiers and accuracies lists
clf2 = []
accuracy.clear()

for i, c in enumerate(C):
    # Apply Linear SVM
    clf2.append(LinearSVC(C=c))
    clf2[i].fit(features_train, target_train)

    # Plot the decision boundaries
    Z = clf2[i].predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("Linear SVM (C = %f)" % c)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(clf2[i].score(features_val, target_val))

plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("Linear SVM accuracy on validation set")
plt.show()

best_C_index_1 = np.argmax(accuracy)
accuracy_test = clf2[best_C_index_1].score(features_test, target_test)
print("Best C is %f and corresponding accuracy on test set is %f%%" % (C[best_C_index_1], accuracy_test * 100))

# %% SVM with RBF kernel
print("\n\t- SVM with RBF kernel")

# initialize classifiers and accuracies lists
clf3 = []
accuracy.clear()

for i, c in enumerate(C):
    # Apply SVM
    clf3.append(SVC(C=c, gamma='auto', kernel='rbf'))
    clf3[i].fit(features_train, target_train)

    # Plot the decision boundaries
    Z = clf3[i].predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("SVM with RBF kernel (C = %f)" % c)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(clf3[i].score(features_val, target_val))

plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("SVM with RBF kernel accuracy on validation set")
plt.show()

best_C_index_2 = np.argmax(accuracy)
accuracy_test = clf3[best_C_index_2].score(features_test, target_test)
print("Best C is %f and corresponding accuracy on test set is %f%%" % (C[best_C_index_2], accuracy_test * 100))

# %% SVM with RBF kernel and grid search
print("\n\t- SVM with RBF kernel and grid search")

gamma_min = -7
gamma_max = -1
n_g = 7

C_min = -3
C_max = 3
n_C = 7

param_grid = {'C': np.logspace(C_min, C_max, n_C), 'gamma': np.logspace(gamma_min, gamma_max, n_g), 'kernel': ['rbf']}
best_score = 0
best_params = {'C': C_min, 'gamma': gamma_min}

for params in ParameterGrid(param_grid):
    model = SVC(**params)
    model.fit(features_train, target_train)
    score = model.score(features_val, target_val)
    if score > best_score:
        best_score = score
        best_params = params
        clf4 = model

# Plot the decision boundaries
Z = clf4.predict(np.c_[x.ravel(), y.ravel()])
Z = Z.reshape(x.shape)
plt.pcolormesh(x, y, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())
plt.title("SVM with RBF kernel (C = %f, gamma = %f)" % (best_params['C'], best_params['gamma']))

plt.show()

accuracy_test = clf4.score(features_test, target_test)
print("Best params are C=%f and gamma=%f and corresponding accuracy on test set is %f%%"
      % (best_params['C'], best_params['gamma'], accuracy_test * 100))

# %% K-Fold
print("\n\t- K-Fold")

features_train = np.concatenate((features_train, features_val))
target_train = np.concatenate((target_train, target_val))


