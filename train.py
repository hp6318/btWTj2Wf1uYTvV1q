import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pickle

# read the file
df = pd.read_csv("./data/ACME-HappinessSurvey2020.csv",delimiter=',')
# print(df[:10])

#Separate the features and labels
XTrain = df[['X1','X2','X3','X4','X5','X6']]
yTrain = df[['Y']]

#train the model

#Logistic Regression
model_LogReg = LogisticRegression(penalty='l1',solver='liblinear').fit(XTrain,yTrain)

#Support Vector CLASSIFIER
model_svClf = LinearSVC(max_iter=1000, tol=1e-3).fit(XTrain,yTrain)

#RandomForest CLASSIFIER
model_randomForest = RandomForestClassifier(n_estimators=50).fit(XTrain,yTrain)

print("Train accuracy | Logistic Regression : ", model_LogReg.score(XTrain,yTrain))
print("Train accuracy | Support Vector Classifier : ", model_svClf.score(XTrain,yTrain))
print("Train accuracy | Random Forest Classifier : ", model_randomForest.score(XTrain,yTrain))

#save the model
pickle.dump(model_randomForest,open("model.pkl",'wb'))

#feature importance
importances = model_randomForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_randomForest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=['X1','X2','X3','X4','X5','X6'])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

result = permutation_importance(model_randomForest, XTrain, yTrain, n_repeats=10, random_state=42, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=['X1','X2','X3','X4','X5','X6'])
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig("Feature_importance.png")
plt.show()
