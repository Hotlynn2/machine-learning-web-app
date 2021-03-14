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
    st.markdown('This is a no-dcode Machine learning interface. Enjoy🥂')
    st.sidebar.markdown('This is a no-dcode Machine learning interface. Enjoy🥂')

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
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.25, random_state = 120)
        return X_train, y_train, X_test, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, X_test, y_test, displat_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_confusion_matrix(model, X_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_confusion_matrix(model, X_test, y_test)
            st.pyplot()

    df = load_df()
    X_train, y_train, X_test, y_test = split(df) 
    class_names = ['edible', 'poisonous']
    






    
    if st.sidebar.checkbox('show raw dataset', False):
        st.subheader('Mushroom Data Set (Classification)')
        st.write(df)

    # if st.sidebar.select_slider('Confusion Matrix', False):
    #     st.subheader('Confusion Matrix Plot')
    #     st.write(metrics)




if __name__ == '__main__':
    main()