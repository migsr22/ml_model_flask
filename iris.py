import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)
#sv = SVC(kernel='linear').fit(X_train,y_train)

pickle.dump(knn, open('iri.pkl', 'wb'))
#pickle.dump(sv, open('iri.pkl', 'wb'))
