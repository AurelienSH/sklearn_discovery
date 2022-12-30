from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def SVM(X_train, X_test, y_train, y_test):
    # Create the SVM model
    svm = SVC()

    # Fit the model to the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    return( svm, f'Test accuracy: {accuracy:.2f}')
    
def NaiveBayes(X_train, X_test, y_train, y_test):
    # Create the Naive Bayes model
    nb = MultinomialNB()

    # Fit the model to the training data
    nb.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    return( nb, f'Test accuracy: {accuracy:.2f}')
    
def LR(X_train, y_train,X_test,y_test):
    # Create the logistic regression model
    lr = LogisticRegression()

    # Fit the model to the training data
    lr.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lr.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    return( lr, f'Test accuracy: {accuracy:.2f}')

def RandomForest(X_train, X_test, y_train, y_test):
    # Create the random forest model
    rf = RandomForestClassifier()

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    return( rf, f'Test accuracy: {accuracy:.2f}')
