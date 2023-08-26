import pandas as pd
from sklearn.metrics import accuracy_score
import pickle


#load the saved model
model = pickle.load(open('model.pkl','rb'))

#load the test file
df = pd.read_csv("./data/ACME-HappinessSurvey2020.csv",delimiter=',') #change the file path

#Separate the features and labels
XTest = df[['X1','X2','X3','X4','X5','X6']]
yTest = df[['Y']]

yTest_predict = model.predict(XTest)

print("Test Accuracy: ",accuracy_score(yTest,yTest_predict))
