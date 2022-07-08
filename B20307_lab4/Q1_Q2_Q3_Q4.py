import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("SteelPlateFaults-2class.csv", encoding='utf-8')
# data of class 0
X_0 = df[df['Class'] == 0]
# class 0
X_label_0 = X_0['Class']
# data of class 1
X_1 = df[df['Class'] == 1]
# class 1
X_label_1 = X_1['Class']
# Splitting data of class 0 into train and test samples
[X_train_0, X_test_0, X_label_train_0, X_label_test_0] = train_test_split(X_0, X_label_0, test_size=0.3,
                                                                          random_state=42,
                                                                          shuffle=True)
# Splitting data of class 1 into train and test samples
[X_train_1, X_test_1, X_label_train_1, X_label_test_1] = train_test_split(X_1, X_label_1, test_size=0.3,
                                                                          random_state=42,
                                                                          shuffle=True)

# train data
train_data = X_train_0.append(X_train_1)
train_data_label = X_label_train_0.append(X_label_train_1)

# test data
test_data = X_test_0.append(X_test_1)
test_data_label = X_label_test_0.append(X_label_test_1)

# saving train and test data into separate data
train_data.to_csv('SteelPlateFaults-train.csv', encoding='utf-8')
test_data.to_csv('SteelPlateFaults-test.csv', encoding='utf-8')

k = [1, 3, 5]
accuracy_k = []
for i in k:
    # k nearest neighbour for k =1,3,5
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_data, train_data_label)  # data fed into model
    predicted_data = knn.predict(test_data)

    # Question1 part a
    # finding confusion matrix
    conf_matrix = confusion_matrix(test_data_label, predicted_data)
    # Question1 part b
    # accuracy
    accuracy = round(accuracy_score(test_data_label, predicted_data), 3)
    accuracy_k.append(accuracy * 100)
    print(f'confusion matrix for k = {i} is: ')
    print(conf_matrix)
    print(f'accuracy for k = {i} is {accuracy * 100}')

# question 2
data_train = train_data.drop('Class', axis=1)
Max = data_train.max()
Min = data_train.min()
# print(Min, Max)
diff = Max - Min
nor_train = (data_train - Min) / diff
print(nor_train)
nor_train.to_csv('SteelPlateFaults-train-Normalised.csv', encoding='utf-8')
data_test = test_data.drop('Class', axis=1)
Max1 = data_test.max()
Min1 = data_test.min()
diff1 = Max1 - Min1
nor_test = (data_test - Min1) / diff1
print(nor_test)
nor_test.to_csv('SteelPlateFaults-test-normalised.csv', encoding='utf-8')

K = [1, 3, 5]
accuracy_k1 = []
for i in k:
    # k nearest neighbour for k=1,3,5
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(nor_train, train_data_label)  # data fed into model
    predicted_data1 = knn1.predict(nor_test)

    # question2 part a
    # finding confusion matrix
    conf_matrix1 = confusion_matrix(test_data_label, predicted_data1)

    # question2 part b
    # accuracy
    accuracy_1 = round(accuracy_score(test_data_label, predicted_data1), 3)
    accuracy_k1.append(accuracy_1 * 100)
    print(f'confusion matrix for k = {i} is: ')
    print(conf_matrix1)
    print(f'accuracy for k = {i} is {accuracy_1 * 100}')

# Question 3
# Reading files
df_train = pd.read_csv('SteelPlateFaults-train.csv')
df_test = pd.read_csv('SteelPlateFaults-test.csv')

# Dropping unnamed column
df_train.drop(df_train.columns[0], axis=1, inplace=True)
df_test.drop(df_test.columns[0], axis=1, inplace=True)

df_train.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1, inplace=True)
df_test.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1, inplace=True)

# Separating data of two classes
train_data_0 = df_train[df_train['Class'] == 0]
train_data_1 = df_train[df_train['Class'] == 1]

test_data_0 = df_test[df_test['Class'] == 0]
test_data_1 = df_test[df_test['Class'] == 1]

# Class labels of each class of trained and test data
train_label_0 = train_data_0['Class']
train_label_1 = train_data_1['Class']

test_label_0 = test_data_0['Class']
test_label_1 = test_data_1['Class']

# Dropping class column to perform bayes classification
train_data_0.drop(['Class'], axis=1, inplace=True)
train_data_1.drop(['Class'], axis=1, inplace=True)
test_data_0.drop(['Class'], axis=1, inplace=True)
test_data_1.drop(['Class'], axis=1, inplace=True)

# Prior probability of class 0
prob_class_0 = len(train_label_0) / (len(train_label_0) + len(train_label_1))

# Prior probability of class 1
prob_class_1 = 1 - prob_class_0

# Mean vectors of trained data of class 0 and 1
mean_vector_0 = train_data_0.mean()
mean_vector_1 = train_data_1.mean()

print('mean vector of class 0 is')
print(round(mean_vector_0, 3))
print('mean vector of class 1 is')
print(round(mean_vector_1, 3))

# Covariance matrices of trained data of class 0 and 1
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
# cov_matrix_0 = train_data_0.cov()
# cov_matrix_1 = train_data_1.cov()

cov_matrix_0 = np.cov(train_data_0.T)
cov_matrix_1 = np.cov(train_data_1.T)

print('covariance matrix of class 0: ', cov_matrix_0)
print('covariance matrix of class 1:', cov_matrix_1)

# Saving covariance matrices as csv files
# cov_matrix_0.to_csv('Covariance matrix of class 0.csv')
# cov_matrix_1.to_csv('Covariance matrix of class 1.csv')
pd.DataFrame(cov_matrix_0).to_csv('Covariance matrix of class 0.csv')
pd.DataFrame(cov_matrix_1).to_csv('Covariance matrix of class 1.csv')
# Train data of two classes
total_train = train_data_0.append(train_data_1)

# test data and class labels of two classes
total_test = np.array(test_data_0.append(test_data_1))
total_test_label = np.array(test_label_0.append(test_label_1))

# list to store predicted class of test data
pred_test_class = []

# Bayes classifier
for i in range(len(total_test)):
    # likelihood = e^(-0.5*((x-mean)T)(cov_matrix^-1)(x-mean))/(2pi^(d/2)(det(cov_matrix)*0.5), d = dimension of data

    # computing likelihood of class 0
    p1_0 = np.dot(np.transpose(total_test[i] - mean_vector_0), np.linalg.inv(cov_matrix_0))
    p2_0 = np.dot(p1_0, (total_test[i] - mean_vector_0))
    lik_0 = np.exp(-0.5 * p2_0) / (((2 * np.pi) ** 11.5) * (np.linalg.det(cov_matrix_0) ** 0.5))

    # computing likelihood of class 1
    p1_1 = np.dot(np.transpose(total_test[i] - mean_vector_1), np.linalg.inv(cov_matrix_1))
    p2_1 = np.dot(p1_1, (total_test[i] - mean_vector_1))
    lik_1 = np.exp(-0.5 * p2_1) / (((2 * np.pi) * 11.5) * (np.linalg.det(cov_matrix_1) * 0.5))

    # Total probability
    p_x = (lik_0 * prob_class_0) + (lik_1 * prob_class_1)

    # Posterior probability of class 0
    prob_c0_x = (lik_0 * prob_class_0) / p_x

    # posterior probability of class 1
    prob_c1_x = (lik_1 * prob_class_1) / p_x

    # assigning class label to test sample
    if prob_c0_x > prob_c1_x:
        pred_class = 0
    else:
        pred_class = 1

    pred_test_class.append(pred_class)

predicted_data = np.array(pred_test_class)
# Finding confusion matrix
confusion_matrix = confusion_matrix(total_test_label, predicted_data)

print('Confusion matrix is')
print(confusion_matrix)

# Finding accuracy
accuracy = round(accuracy_score(total_test_label, predicted_data), 3)
print(f'Accuracy in percentage is {round((accuracy * 100), 3)}')

# Question 4
# Tabulating the best results of all three classifiers
comparison = {"S. No.": [1, 2, 3], "Classifier": ["KNN", "KNN on normalised data", "Bayes"],
              "Accuracy (in %)": [round(100 * accuracy, 3), round(100 * accuracy_1, 3),
                                  round(100 * accuracy_score(total_test_label, predicted_data), 3)]}

table = pd.DataFrame(comparison)
print("Comparison between classifiers based upon classification accuracy:\n", table)
