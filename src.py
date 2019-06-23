import numpy as np
import pandas as pd
import re
import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import preprocessing

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def prepare_data(df):
    for x in range(len(df)):
        for y in range(22):
            if type(df[x][y]) is str:
                df[x][y] = df[x][y].replace('"', "")
                df[x][y] = df[x][y].replace('.', "")


    for x in range(len(df)):
        for y in range(22):
            if type(df[x][y]) is str and hasNumbers(df[x][y]):
                df[x][y] = float(df[x][y])
    df = pd.DataFrame(df)
    df.dropna(inplace=True, subset=[21])
    df = df.drop([19],1)
    df = df.to_numpy()

    for x in range(len(df)):
        for y in range(20,21):
            if df[x][y] == 'yes':
                df[x][y] = 1
            elif df[x][y] == 'no':
                df[x][y] = 0

    df = pd.DataFrame(df)
    return df


pp = pprint.PrettyPrinter(indent=4)
training_df = pd.read_csv('training_new.csv')
validation_df = pd.read_csv('validation_new.csv')
training_df = training_df.to_numpy()
validation_df = validation_df.to_numpy()
df1 = prepare_data(training_df)
df2 = prepare_data(validation_df)
df = pd.concat([df1,df2], axis = 0)


df = pd.get_dummies(df, columns=[0,5,6,7,8,11,12,14,15])
df.fillna(df.mean(), inplace= True)
label = df[20]
training_label = label[:df1.shape[0]]
validation_label = label[df1.shape[0]+1:]
features = df.drop([20],1)
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
features = pd.DataFrame(features)
training_features = features[:df1.shape[0]]
validation_features = features[df1.shape[0]+1:]

#clf = MultinomialNB()
#clf = LogisticRegression(random_state=0, solver='lbfgs')
clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10,10), random_state=1, max_iter = 200)
#clf = SVC(gamma='auto')
#clf = tree.DecisionTreeClassifier()
clf.fit(training_features, training_label)
#clf.fit(x_train, y_train)

y_predicted = clf.predict(validation_features)
#y_predicted = clf.predict(x_test)

accuracy = accuracy_score(validation_label, y_predicted)
precision = precision_score(validation_label, y_predicted)
recall = recall_score(validation_label, y_predicted)

# accuracy = accuracy_score(y_test, y_predicted) # excellent results for all metrics when splitting and testing on the training set
# precision = precision_score(y_test, y_predicted)
# recall = recall_score(y_test, y_predicted)

print("accuracy:" + str(accuracy))
print("precision:" + str(precision))
print("recall:" + str(recall))
print("f1-score:" + str(2*(precision*recall/(precision + recall))))
print(classification_report(validation_label,y_predicted))
