print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets, svm, metrics

# The digits dataset
train=pd.read_csv('./train.csv')
data_test=pd.read_csv('./test.csv')
data_train=train.iloc[:,1:].values
label_train=train.iloc[:,0].values

data_test=data_test.values
	
data_train[data_train>0]=1
data_test[data_test>0]=1

print(data_train)
print(label_train)
# Create a classifier: a support vector classifier
classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
#classifier = svm.SVC()


# Learn the digits on the first half of the digits
classifier.fit(data_train, label_train)

# Predict the value of the digit on the second half:
#expected = label_test
predicted = classifier.predict(data_test)
#print(expected)
df=pd.DataFrame(predicted)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv',header=True)
print(predicted)
