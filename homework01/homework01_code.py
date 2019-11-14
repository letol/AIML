import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

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

# initialize accuracies list
accuracy = []

for i, k in enumerate(K):
    # Apply k-NN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features_train, target_train)

    # Plot the decision boundaries
    Z = model.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("K-Nearest Neighbor (k = %i)" % k)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(model.score(features_val, target_val))

plt.ylim(0.5, 1)
plt.ylabel("accuracy")
plt.xlabel("K")
plt.plot(K, accuracy, '-', K, accuracy, 'o')
plt.title("k-NN accuracy on validation set")
plt.show()

best_K_index = np.argmax(accuracy)
clf1 = KNeighborsClassifier(n_neighbors=K[best_K_index])
clf1.fit(features_train_val, target_train_val)
accuracy_test = clf1.score(features_test, target_test)
print("Best K is %i and corresponding accuracy on test set is %f%%" % (K[best_K_index], accuracy_test * 100))

# %% Linear SVM
print("\n\t- Linear SVM")

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# initialize accuracies list
accuracy.clear()

for i, c in enumerate(C):
    # Apply Linear SVM
    model = LinearSVC(C=c)
    model.fit(features_train, target_train)

    # Plot the decision boundaries
    Z = model.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("Linear SVM (C = %f)" % c)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(model.score(features_val, target_val))

plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("Linear SVM accuracy on validation set")
plt.show()

best_C_index_1 = np.argmax(accuracy)
clf2 = LinearSVC(C=C[best_C_index_1])
clf2.fit(features_train_val, target_train_val)
accuracy_test = clf2.score(features_test, target_test)
print("Best C is %f and corresponding accuracy on test set is %f%%" % (C[best_C_index_1], accuracy_test * 100))

# %% SVM with RBF kernel
print("\n\t- SVM with RBF kernel")

# initialize accuracies list
accuracy.clear()

for i, c in enumerate(C):
    # Apply SVM
    model = SVC(C=c, gamma='auto', kernel='rbf')
    model.fit(features_train, target_train)

    # Plot the decision boundaries
    Z = model.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.pcolormesh(x, y, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("SVM with RBF kernel (C = %f)" % c)

    plt.show()

    # Evaluate on the validation set
    accuracy.append(model.score(features_val, target_val))

plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.semilogx(C, accuracy, '-', C, accuracy, 'o')
plt.title("SVM with RBF kernel accuracy on validation set")
plt.show()

best_C_index_2 = np.argmax(accuracy)
clf3 = SVC(C=C[best_C_index_2], gamma='auto', kernel='rbf')
clf3.fit(features_train_val, target_train_val)
accuracy_test = clf3.score(features_test, target_test)
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
best_params = {'C': C_min, 'gamma': gamma_min, 'kernel': 'rbf'}

for params in ParameterGrid(param_grid):
    model = SVC(**params)
    model.fit(features_train, target_train)
    score = model.score(features_val, target_val)
    if score > best_score:
        best_score = score
        best_params = params

# Plot the decision boundaries
clf4 = SVC(**best_params)
clf4.fit(features_train_val, target_train_val)
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

kf = KFold(n_splits=5)
clf5 = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=kf.split(features_train_val), iid=False)
clf5.fit(features_train_val, target_train_val)
best_params = clf5.best_params_

# Plot the decision boundaries
Z = clf5.predict(np.c_[x.ravel(), y.ravel()])
Z = Z.reshape(x.shape)
plt.pcolormesh(x, y, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())
plt.title("SVM with RBF kernel and K-Fold (C = %f, gamma = %f)" % (best_params['C'], best_params['gamma']))

plt.show()

accuracy_test = clf5.score(features_test, target_test)
print("Best params are C=%f and gamma=%f and corresponding accuracy on test set is %f%%"
      % (best_params['C'], best_params['gamma'], accuracy_test * 100))
