# importing all requored libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import glob

#function for training the model
def training_model(features,labels):
    model = svm.SVC(C=1000, gamma=0.0001, kernel='rbf')
    model.fit(features,labels)
    return model

#loading the data into pandas dataframe from the csv file
train_data = pd.read_csv('spam_classifier_training_data')
emails = train_data["text"]
labels = train_data["label"]

#converting mails into sparse matrix of <class 'numpy.int64'> to extract features
cv = CountVectorizer()
Extracted_features = cv.fit_transform(emails)

#function call for training the model using Support Vector Machine
model = training_model(Extracted_features,labels)

#function which outputs the 'output.txt' file which contains class of each mail in test dataset
def predict_spam_or_ham(model):

    #gets the path of all emails present in the current folder
    txt_files = glob.glob("test/*.txt")


    prediction = []         # Vector for storing the predicted value of each mail file

    # looping over all the files one by one
    for i in txt_files:
        with open(i, 'r') as fd:
            email = fd.read()       #reads an email
            email_features = cv.transform([email])      #feature extraction of test emails
            prediction.append(model.predict(email_features)[0])     #prediction using trained model


    # storing the predicted values into 'output.txt' file
    with open('output.txt', mode='w') as fd:
        for i in prediction:
            fd.writelines(str(i) + '\n')

# function call for the procedure which when invoked, outputs a file of predicted values.
predict_spam_or_ham(model)