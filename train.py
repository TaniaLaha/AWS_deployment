import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

os.makedirs("model", exist_ok=True)
with open("model/iris_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model trained and saved to model/iris_model.pkl")