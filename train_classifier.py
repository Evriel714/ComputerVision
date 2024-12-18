import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
test_data_dict = pickle.load(open('./test_data.pickle', 'rb'))

x_train = np.asarray(data_dict['data'])
y_train = np.asarray(data_dict['labels'])
x_test = np.asarray(test_data_dict['data'])
y_test = np.asarray(test_data_dict['labels'])

# model = KNeighborsClassifier(metric='euclidean', n_neighbors=5, p=1, weights='distance')
model = KNeighborsClassifier()
# model = RandomForestClassifier()
# model = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=200)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# clf = RandomForestClassifier(random_state=42)

# param_grid = {
#     'n_estimators': [50, 100, 200],        # Number of trees
#     'max_depth': [10, 20, None],          # Tree depth
#     'min_samples_split': [2, 5, 10],      # Min samples required to split a node
#     'min_samples_leaf': [1, 2, 4],        # Min samples at a leaf node
#     'max_features': ['sqrt', 'log2'],     # Number of features considered for splits
# }

# # Set up Grid Search
# grid_search = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=5,                       # 5-fold cross-validation
#     scoring='accuracy',         # Optimize for accuracy
#     n_jobs=-1,                  # Use all available CPU cores
#     verbose=2                   # Print progress
# )

# # Fit the model
# grid_search.fit(x_train, y_train)

# # Display the best parameters and accuracy
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation accuracy:", grid_search.best_score_)

# # Test the best model on the test set
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# print("Test set accuracy:", accuracy_score(y_test, y_pred))

# knn = KNeighborsClassifier()

# # Define the parameter grid
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],      # Number of neighbors to use
#     'weights': ['uniform', 'distance'],   # Weighting function
#     'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
#     'p': [1, 2],  # Power parameter for Minkowski (1 = Manhattan, 2 = Euclidean)
# }

# # Set up Grid Search
# grid_search = GridSearchCV(
#     estimator=knn,
#     param_grid=param_grid,
#     cv=5,                       # 5-fold cross-validation
#     scoring='accuracy',         # Optimize for accuracy
#     n_jobs=-1,                  # Use all available CPU cores
#     verbose=2                   # Print progress
# )

# # Fit the model
# grid_search.fit(x_train, y_train)

# # Display the best parameters and accuracy
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation accuracy:", grid_search.best_score_)

# # Test the best model on the test set
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# print("Test set accuracy:", accuracy_score(y_test, y_pred))