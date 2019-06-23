import numpy as np
import pandas as pd
import re
import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import preprocessing

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

pp = pprint.PrettyPrinter(indent=4)
df = pd.read_csv('training_new.csv')
df = df.to_numpy()


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
print(df)
df.dropna(inplace=True, subset=[21])
df = df.drop([19],1)
print(df)
df = df.to_numpy()
count_yes = 0
count_no = 0 
for x in range(len(df)):
    for y in range(20,21):
        if df[x][y] == 'yes':
            df[x][y] = 1
            count_yes += 1
        elif df[x][y] == 'no':
            df[x][y] = 0
            count_no += 1

print(count_no)
print(count_yes)

df = pd.DataFrame(df)
df = pd.get_dummies(df, columns=[0,5,6,7,8,11,12,14,15])
df.fillna(df.mean(), inplace= True)

pp.pprint(df)
label = df[20]
features = df.drop([20],1)

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
print(features)
features = pd.DataFrame(features)
#features = preprocessing.normalize(features)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

#X = pd.concat([x_train, y_train], axis=1)

# separate minority and majority classes
#yes = X[df[20]==1]
#no = X[df[20]==0]


# upsample minority
#no_upsampled = resample(no,
#                          replace=True, # sample with replacement
#                          n_samples=len(yes), # match number in majority class
#                         random_state=27) # reproducible results

# combine majority and upsampled minority
#upsampled = pd.concat([yes, no_upsampled])

#y_train = upsampled[20]
#x_train = upsampled.drop(20, axis=1)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16,10))
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
print(accuracy)
#print(confusion_matrix(y_test,y_predicted))
print(precision)
print(recall)
print(2*(precision*recall/(precision + recall)))
print(classification_report(y_test,y_predicted))
