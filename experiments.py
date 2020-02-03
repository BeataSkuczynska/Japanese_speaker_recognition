import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import read_data, check_timeseries_length
from utils import MidpointNormalize

aeTrain = np.genfromtxt("resources/ae.train", dtype=None)
aeTest = np.genfromtxt("resources/ae.test", dtype=None)

train_lengths, train_maxy = check_timeseries_length(aeTrain)
test_lengths, test_maxy = check_timeseries_length(aeTest)
maxy = max(train_maxy, test_maxy)
train_input, train_output = read_data(aeTrain, train_lengths, 270, maxy)
test_input, test_output = read_data(aeTest, test_lengths, 370, maxy, test=True)


'''C-Support Vector Classification with linear kernel and mean of the timeseries over each channel '''
# transformed_train = np.zeros(shape=(270, 12))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#     transformed_train[c] = np.around(np.mean(train_input[c, :], axis=0), 4)
#
# clf = svm.SVC(kernel='linear')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 85% validation score'''

'''C-Support Vector Classification with rbf kernel and mean of the timeseries over each channel '''
# transformed_train = np.zeros(shape=(270, 12))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#     transformed_train[c] = np.around(np.mean(train_input[c, :], axis=0), 4)
#
# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 88% validation score'''

'''C-Support Vector Classification with sigmoid kernel and mean of the timeseries over each channel '''
# transformed_train = np.zeros(shape=(270, 12))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#     transformed_train[c] = np.around(np.mean(train_input[c, :], axis=0), 4)
#
# clf = svm.SVC(kernel='sigmoid')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 67% validation score'''

'''C-Support Vector Classification with poly kernel and mean of the timeseries over each channel '''
# transformed_train = np.zeros(shape=(270, 12))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#     transformed_train[c] = np.around(np.mean(train_input[c, :], axis=0), 4)
#
# clf = svm.SVC(kernel='poly')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 72% validation score'''


'''C-Support Vector Classification with rbf kernel, modified regularization 
and mean of the timeseries over each channel '''
# transformed_train = np.zeros(shape=(270, 12))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#     transformed_train[c] = np.around(np.mean(train_input[c, :], axis=0), 4)
#
# for c_param in np.linspace(0.5, 10.0, 20):
#     clf = svm.SVC(C=c_param, kernel='rbf')
#     scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
#     # print(scores)
#     print("For C parameter ", str(c_param), ": ", round(scores.mean(), 3))
'''The highest obtained score was around 89%. Generally, not worth it.'''

'''C-Support Vector Classification with linear kernel and flattened cepstral data'''
# transformed_train = np.reshape(train_input, (270, -1))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# clf = svm.SVC(kernel='linear')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 89% validation score'''


'''C-Support Vector Classification with rbf kernel and flattened cepstral data'''
# transformed_train = np.reshape(train_input, (270, -1))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 90% validation score'''

'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds'''
# transformed_train = np.reshape(train_input, (270, -1))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=10)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 91.5% validation score'''

'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds. 
Trained on the full train dataset, tested on test data'''
# transformed_train = np.reshape(train_input, (270, -1))
# transformed_test = np.reshape(test_input, (370, -1))
# train_output_simple = np.zeros(shape=270)
# test_output_simple = np.zeros(shape=370)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# for c in range(370):
#     speaker = np.nonzero(test_output[c, 0])[0]
#     test_output_simple[c] = speaker
#
# clf = svm.SVC(kernel='rbf')
# clf.fit(transformed_train, train_output_simple)
# score = clf.score(transformed_test, test_output_simple)
# print(score)
'''Around 95.7% validation score'''

'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds. 
Features scaled to [0, 1] range'''
# min_max_scaler = preprocessing.MinMaxScaler()
#
# transformed_train = np.reshape(train_input, (270, -1))
# transformed_train = min_max_scaler.fit_transform(transformed_train)
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, transformed_train, train_output_simple, cv=5)
# print(scores)
# print(round(scores.mean(), 3), round(scores.std(), 3))
'''Around 90.5% validation score'''

'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds. 
Features scaled to [0, 1] range. Initial grid search over C and gamma parameters.'''
# min_max_scaler = preprocessing.MinMaxScaler()
#
# transformed_train = np.reshape(train_input, (270, -1))
# transformed_train = min_max_scaler.fit_transform(transformed_train)
# print (np.var(transformed_train))
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# params = dict(C=np.logspace(-3, 3, 7), gamma=np.logspace(-3, 3, 7))
# print(params)
# clf = svm.SVC(kernel='rbf')
# grid_search = GridSearchCV(clf, params, cv=5, verbose=1)
# grid_search.fit(transformed_train, train_output_simple)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
'''{'C': 100.0, 'gamma': 0.001}
            94.1%'''

'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds. 
Features scaled to [0, 1] range. Further grid search over C and gamma parameters.'''
# min_max_scaler = preprocessing.MinMaxScaler()
#
# transformed_train = np.reshape(train_input, (270, -1))
# transformed_train = min_max_scaler.fit_transform(transformed_train)
# train_output_simple = np.zeros(shape=270)
#
# for c in range(270):
#     speaker = np.nonzero(train_output[c, 0])[0]
#     train_output_simple[c] = speaker
#
# C_range = np.arange(10, 120, 10)
# gamma_range = np.linspace(0.0005, 0.0015, 11)
# params = dict(C=C_range, gamma=gamma_range)
#
# clf = svm.SVC(kernel='rbf')
# grid_search = GridSearchCV(clf, params, cv=5, verbose=1)
# grid_search.fit(transformed_train, train_output_simple)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
'''{'C': 40.0, 'gamma': 0.0014}
            0.944'''

'''Plot of the heatmap of validation accuracy'''
# scores = grid_search.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
#            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# gamma_range_list = list(gamma_range)
# gamma_range_rounded = [round(i, 5) for i in gamma_range_list]
# plt.xticks(np.arange(len(gamma_range)), gamma_range_rounded, rotation=50)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.title('Validation accuracy')
# plt.show()


'''C-Support Vector Classification with rbf kernel, flattened cepstral data and increased amount of folds. 
Features scaled to [0, 1] range. C and gamma parameters chosen from grid search. Trained on a full dataset.'''
transformed_train = np.reshape(train_input, (270, -1))
transformed_test = np.reshape(test_input, (370, -1))
train_output_simple = np.zeros(shape=270)
test_output_simple = np.zeros(shape=370)

for c in range(270):
    speaker = np.nonzero(train_output[c, 0])[0]
    train_output_simple[c] = speaker

for c in range(370):
    speaker = np.nonzero(test_output[c, 0])[0]
    test_output_simple[c] = speaker

clf = svm.SVC(kernel='rbf', C=40.0, gamma=0.0014)
clf.fit(transformed_train, train_output_simple)
predicted = clf.predict(transformed_test)
score = clf.score(transformed_test, test_output_simple)
print(score)
'''96.2% accuracy'''

'''Confusion matrix for final model'''
conf_matrix = confusion_matrix(test_output_simple, predicted, normalize="true")
conf_matrix_rounded = np.around(conf_matrix, 3)
print(conf_matrix_rounded)
