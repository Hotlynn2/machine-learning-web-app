import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title('Machine Learning Web Application')
    st.sidebar.title('Machine Learning Web Application')
    st.markdown('This is a low-code Machine learning interface. Enjoy🥂')
    st.sidebar.markdown('This is a low-code Machine learning interface. Enjoy🥂')

    @st.cache(persist=True)
    def load_df():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist = True)
    def split(dataframe):
        X = dataframe.drop('type', axis = 1)
        y = dataframe.type
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
        return X_train, y_train, X_test, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, X_test, y_test, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
        

    df = load_df()
    X_train, y_train, X_test, y_test = split(df) 
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Choose Classifier", ("Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"))


    if st.sidebar.checkbox('show raw dataset', False):
        st.subheader('Mushroom Data Set (Classification)')
        st.write(df)

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader('SVM Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization Number)", 0.01, 10.0, step = 0.01, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf","linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma Kernel Coefficient", ("scale", "auto"), key = 'gamma' )

        metrics = st.sidebar.multiselect('What metric should be used', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C = C, gamma = gamma, kernel = kernel)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write('Accuracy :', model.score(X_test, y_test).round(2))
            st.write('Precision :', precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write('Recall :', recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Logistic Regression Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
        max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, key = 'max_iter')

        metrics = st.sidebar.multiselect('What metric should be used', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write('Accuracy :', model.score(X_test, y_test).round(2))
            st.write('Precision :', precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write('Recall :', recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader('Random Forest Model Hyperparameters')
        n_estimators = st.sidebar.slider("Number of trees in the forest", 10, 100, key = 'n_estimators')
        criterion = st.sidebar.radio("Criterion", ("gini","entropy"), key = 'criterion')
        max_depth = st.sidebar.number_input("Maximum depth of a tree", 1, 50, step = 1, key = 'max_depth')

        metrics = st.sidebar.multiselect('What metric should be used', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key = 'classify'):
            st.subheader('Random Forest Results')
            model = LogisticRegression(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write('Accuracy :', model.score(X_test, y_test).round(2))
            st.write('Precision :', precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write('Recall :', recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
        


 

if __name__ == '__main__':
    main()