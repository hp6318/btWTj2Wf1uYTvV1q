import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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


