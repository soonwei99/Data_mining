import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from kmodes.kmodes import KModes

from sklearn.model_selection import GridSearchCV
import imblearn

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Laundry_Data.csv')
cleandf = pd.read_csv('export_dataframe.csv')



cleandf['Time'] = cleandf['Time'].str.replace(';',':')
cleandf['Time'] = pd.to_datetime(cleandf['Time'], format='%H:%M:%S').dt.hour
cleandf['Date'] = pd.to_datetime(cleandf['Date'])
cleandf['Date'] = cleandf['Date'].dt.day_name()
cleandf['Dryer'] = pd.cut(cleandf['Dryer_No'],4,labels=['seven','eight','nine','ten'])
cleandf['Washer'] = pd.cut(cleandf['Washer_No'],4,labels=['three','four','five','six'])
cleandf['Age_Category'] = pd.cut(cleandf['Age_Range'],3,labels=["26-35", "36-45", "46-55"])
cleandf['Time_Category'] = pd.cut(cleandf['Time'],4,labels=['mid_night','morning','afternoon','night'])

st.set_option('deprecation.showPyplotGlobalUse', False)
option = st.sidebar.selectbox(
    'Which number do you like best?', ('Data Mining Project', 'Feature Selection' , 'Classification Modeling' ,'Clustering'))
     
 
if (option ==  'Data Mining Project'):

    st.write("""
    # DATA MINING PROJECT
    """)

    st.write("""
    ## A View Of The Dataset (First 10 row)
    """)
    st.dataframe(df.head(10))

    st.write("""
    ## Data Prepossessing
    """)

    option = st.selectbox(
         'Null Values In Data',
         ('Before', 'After'))
         
    st.write('You selected:', option)
         
    if (option == 'Before'):
        df.isna().sum().plot(kind = 'bar'), plt.title('Columns With Null Data')
        st.pyplot()

    elif(option == 'After'):
        cleandf.isna().sum().plot(kind = 'bar'), plt.title('After Cleaning')
        st.pyplot()
       
        
    st.write("""
    ## Exploratory of Data Analysis
    """)

    option = st.selectbox(
         'What would you like to view from the data?',
         ('Time', 'Age_Category', 'With_Kids' , 'Race' , 'Gender' , 'Date' ))

    st.write('You selected:', option)



    if (option == 'Time'):
        cleandf['Time'].value_counts().plot(kind = 'bar'), plt.title('Time of customer visit to launderettes')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    elif (option == 'Age_Category'):
        cleandf['Age_Category'].value_counts().plot(kind = 'bar'), plt.title('Age Range of customer')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
       
    elif (option == 'With_Kids'):
        cleandf['With_Kids'].value_counts().plot(kind = 'bar'), plt.title('Do customer bring kids?')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    elif (option == 'Race'):
        cleandf['Race'].value_counts().plot(kind = 'bar'), plt.title('Race of customer')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    elif (option == 'Gender'):
        cleandf['Gender'].value_counts().plot(kind = 'bar'), plt.title('Gender of customer')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    elif (option == 'Date'):
        cleandf['Date'].value_counts().plot(kind = 'bar'), plt.title('Which day customer are most likey to come')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    option = st.selectbox(
         'What Categories would you like to view from the data?',
         ('Time vs Race', 'Day vs Time', 'Race vs Basket Size' , 'Time vs Age' , 'Time vs Gender' , 'Time vs Kids' , 'Kids vs Gender', 'BasketSize vs Kids' ))

    st.write('You selected:', option)

    if (option == 'Time vs Race'):
            table = pd.crosstab(cleandf['Time_Category'], cleandf['Race'])
            table.plot(kind = 'bar'), plt.title('Time vs Race'), plt.legend(bbox_to_anchor=(1.05, 1))
            st.pyplot()

    elif (option == 'Day vs Time'):
            table = pd.crosstab(cleandf['Date'], cleandf['Time_Category'])
            table.plot(kind = 'bar'), plt.title('Day vs Time'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'Race vs Basket Size'):
            table = pd.crosstab(cleandf['Race'], cleandf['Basket_Size'])
            table.plot(kind = 'bar'), plt.title('Race vs Basket Size'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'Time vs Age'):
            table = pd.crosstab(cleandf['Time_Category'], cleandf['Age_Category'])
            table.plot(kind = 'bar'), plt.title('Time vs Age'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'Time vs Gender'):
            table = pd.crosstab(cleandf['Time_Category'], cleandf['Gender'])
            table.plot(kind = 'bar'), plt.title('Time vs Gender'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'Time vs Kids'):
            table = pd.crosstab(cleandf['Time_Category'], cleandf['With_Kids'])
            table.plot(kind = 'bar'), plt.title('Time vs Kids'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'Kids vs Gender'):
            table = pd.crosstab(cleandf['With_Kids'], cleandf['Gender'])
            table.plot(kind = 'bar'), plt.title('Kids vs Gender'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()

    elif (option == 'BasketSize vs Kids'):
            table = pd.crosstab(cleandf['Basket_Size'], cleandf['With_Kids'])
            table.plot(kind = 'bar'), plt.title('BasketSize vs Kids'), plt.legend(bbox_to_anchor=(1.35, 1))
            st.pyplot()
     





elif ( option == 'Feature Selection'):

    st.write("""
    ## Feature Selection
    """)

    st.write("""
    ### Boruta Features
    """)
    
    df_dm = cleandf[['Time_Category', 'Date', 'Race','Gender', 'Body_Size', 'Age_Range',
       'With_Kids', 'Kids_Category', 'Basket_Size',  'Attire','shirt_type','pants_type', 'Wash_Item',
       'Spectacles', 'Dryer_No', 'Washer_No']].copy()
           
    df_dm = df_dm[df_dm['Basket_Size'].isin(['big','small'])]
    
    
    df_dm = pd.get_dummies(data = df_dm, columns=['Race','Attire','Kids_Category','Gender','Wash_Item'])
    
    le = LabelEncoder()

    df_dm['Time_Category'] = le.fit_transform(df_dm['Time_Category'])
    col_name = df_dm.columns
    for col in col_name:
        if (df_dm[col].dtypes==object):
            df_dm[col] = le.fit_transform(df_dm[col])
        
        
    y = df_dm['Basket_Size']
    X = df_dm.drop('Basket_Size',axis='columns')

    colnames = X.columns
    
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))
        


    rf = RandomForestClassifier(n_jobs = -1, class_weight="balanced", max_depth = 5)
    feat_selector = BorutaPy(rf, n_estimators = "auto", random_state = 1)
    feat_selector.fit(X.values, y.values.ravel())

    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    
    boruta_score = boruta_score.sort_values("Score", ascending = False)

     
    option = st.selectbox(
         'Choose The Top/Bottom 10 Features',
         ('Top10', 'Bottom10'))

    st.write('You selected:', option)

    if(option == 'Top10'):
        st.dataframe(boruta_score.head(10))

    elif (option == 'Bottom10'):
        st.dataframe(boruta_score.tail(10))
        
    
    st.write('')
    st.write('')
    sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm'),plt.title("Boruta Top 10 Features")
    st.pyplot()


elif (option ==  'Classification Modeling'):
    df_dm = cleandf[['Time_Category', 'Date', 'Race','Gender', 'Body_Size', 'Age_Range',
       'With_Kids', 'Kids_Category', 'Basket_Size',  'Attire','shirt_type','pants_type', 'Wash_Item',
       'Spectacles', 'Dryer_No', 'Washer_No']].copy()
           
    df_dm = df_dm[df_dm['Basket_Size'].isin(['big','small'])]
    
    
    df_dm = pd.get_dummies(data = df_dm, columns=['Race','Attire','Kids_Category','Gender','Wash_Item'])
    
    le = LabelEncoder()

    df_dm['Time_Category'] = le.fit_transform(df_dm['Time_Category'])
    col_name = df_dm.columns
    for col in col_name:
        if (df_dm[col].dtypes==object):
            df_dm[col] = le.fit_transform(df_dm[col])
        
        
    y = df_dm['Basket_Size']
    X = df_dm.drop('Basket_Size',axis='columns')
    
    X = X.drop(['Attire_UNKNOWN', 'Wash_Item_UNKNOWN','Race_UNKNOWN'],axis='columns')
    
    smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=1)
    X_res, y_res = smt.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.3, random_state = 10)

    colnames = X.columns
    option = st.selectbox(
         'Select A Model (With Smote)',
         ('Naives Bayes', 'Random Forest Classifier','Support Vector Machine Classifier','Overview on three classifier above'))

    st.write('You selected:', option)

    if(option == 'Naives Bayes'):
        loaded_model = pickle.load(open('nb_model.sav', 'rb'))
        y_pred = loaded_model.predict(X_test)
        st.write("Accuracy on training set: {:.3f}".format(loaded_model.score(X_train, y_train)))
        st.write("Accuracy on test set: {:.3f}".format(loaded_model.score(X_test, y_test)))
        
        confusion_majority=confusion_matrix(y_test, y_pred)

        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

        st.write('**********************')
        st.write('Mjority TN= ', confusion_majority[0][0])
        st.write('Mjority FP=', confusion_majority[0][1])
        st.write('Mjority FN= ', confusion_majority[1][0])
        st.write('Mjority TP= ', confusion_majority[1][1])
        st.write('**********************')

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

        prob_NB = loaded_model.predict_proba(X_test)
        prob_NB = prob_NB[:,1]


        fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB) 

        auc_NB = roc_auc_score(y_test, prob_NB)
        st.write("AUC: %.2f"%auc_NB)
        
        st.write('**********************')
        fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB)
        plt.plot(fpr_NB, tpr_NB, color='orange', label='NB')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot()
        #st.write(result)
        
    elif(option == 'Random Forest Classifier'):
        loaded_model = pickle.load(open('rf_model.sav', 'rb'))
        y_pred = loaded_model.predict(X_test)
        st.write("Accuracy on training set: {:.3f}".format(loaded_model.score(X_train, y_train)))
        st.write("Accuracy on test set: {:.3f}".format(loaded_model.score(X_test, y_test)))
        
        confusion_majority=confusion_matrix(y_test, y_pred)

        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

        st.write('**********************')
        st.write('Mjority TN= ', confusion_majority[0][0])
        st.write('Mjority FP=', confusion_majority[0][1])
        st.write('Mjority FN= ', confusion_majority[1][0])
        st.write('Mjority TP= ', confusion_majority[1][1])
        st.write('**********************')

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

        prob_rf = loaded_model.predict_proba(X_test)
        prob_rf = prob_rf[:,1]


        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, prob_rf) 

        auc_rf = roc_auc_score(y_test, prob_rf)
        st.write("AUC: %.2f"%auc_rf)
        
        st.write('**********************')
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, prob_rf)
        plt.plot(fpr_rf, tpr_rf, color='blue', label='rf')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot()
        
    elif(option == 'Support Vector Machine Classifier'):
        loaded_model = pickle.load(open('svc_model.sav', 'rb'))
        y_pred = loaded_model.predict(X_test)
        st.write("Accuracy on training set: {:.3f}".format(loaded_model.score(X_train, y_train)))
        st.write("Accuracy on test set: {:.3f}".format(loaded_model.score(X_test, y_test)))
        
        confusion_majority=confusion_matrix(y_test, y_pred)

        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

        st.write('**********************')
        st.write('Mjority TN= ', confusion_majority[0][0])
        st.write('Mjority FP=', confusion_majority[0][1])
        st.write('Mjority FN= ', confusion_majority[1][0])
        st.write('Mjority TP= ', confusion_majority[1][1])
        st.write('**********************')

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

        prob_svm = loaded_model.predict_proba(X_test)
        prob_svm = prob_svm[:,1]


        fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, prob_svm) 

        auc_svm = roc_auc_score(y_test, prob_svm)
        st.write("AUC: %.2f"%auc_svm)
        
        plt.plot(fpr_svm, tpr_svm, color='red', label='svm')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot()
        
    if(option == 'Overview on three classifier above'):
    
        loaded_model1 = pickle.load(open('svc_model.sav', 'rb'))
        prob_svm = loaded_model1.predict_proba(X_test)
        prob_svm = prob_svm[:,1]

        loaded_model2 = pickle.load(open('nb_model.sav', 'rb'))
        prob_nb = loaded_model2.predict_proba(X_test)
        prob_nb = prob_nb[:,1]
        
        loaded_model3 = pickle.load(open('rf_model.sav', 'rb'))
        prob_rf = loaded_model3.predict_proba(X_test)
        prob_rf = prob_rf[:,1]


        fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_nb) 
        fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_rf) 
        fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, prob_svm) 

        plt.plot(fpr_NB, tpr_NB, color='orange', label='NB') 
        plt.plot(fpr_RF, tpr_RF, color='blue', label='RF')  
        plt.plot(fpr_SVM, tpr_SVM, color='red', label='SVM')
        
        st.write('**********************')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



elif (option == 'Clustering'):

    st.write(""" ## Clustering """)
    st.write(""" ### K Modes Clustering """)
    df_clus = cleandf.copy()

    df_clus = cleandf.drop(columns=['No', 'Age_Range' , 'Time' , 'Washer_No', 'Dryer_No'])

    df_clus_copy = df_clus.copy()
    
    
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df_clus = df_clus.apply(le.fit_transform)
  

    option = st.selectbox(
         'Select Number Of Cluster',
         ('1', '2' , '3'))
         
    st.write('You selected:', option)
            
    
    if (option == '1'):
        km_huang = KModes(n_clusters = 1, init = "Huang", n_init = 1, verbose=1)
        fitClusters_huang = km_huang.fit_predict(df_clus)
        
        df_clus = df_clus_copy.reset_index()

        clustersDf = pd.DataFrame(fitClusters_huang)
        clustersDf.columns = ['cluster_predicted']
        combinedDf = pd.concat([df_clus, clustersDf], axis = 1).reset_index()
        combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

        cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
        cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
        
        for col in combinedDf:
            plt.subplots(figsize = (10,5))
            sns.countplot(x='cluster_predicted',hue=col, data = combinedDf)
            st.pyplot()
            st.write('')
            st.write('') 
            
            
    elif (option == '2'):
        km_huang = KModes(n_clusters = 2, init = "Huang", n_init = 1, verbose=1)
        fitClusters_huang = km_huang.fit_predict(df_clus)
        
        df_clus = df_clus_copy.reset_index()

        clustersDf = pd.DataFrame(fitClusters_huang)
        clustersDf.columns = ['cluster_predicted']
        combinedDf = pd.concat([df_clus, clustersDf], axis = 1).reset_index()
        combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

        cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
        cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
        
        for col in combinedDf:
            plt.subplots(figsize = (10,5))
            sns.countplot(x='cluster_predicted',hue=col, data = combinedDf)
            st.pyplot()
            st.write('')
            st.write('') 
            
    elif (option == '3'):
        km_huang = KModes(n_clusters = 3, init = "Huang", n_init = 1, verbose=1)
        fitClusters_huang = km_huang.fit_predict(df_clus)
        
        df_clus = df_clus_copy.reset_index()

        clustersDf = pd.DataFrame(fitClusters_huang)
        clustersDf.columns = ['cluster_predicted']
        combinedDf = pd.concat([df_clus, clustersDf], axis = 1).reset_index()
        combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

        cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
        cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
        
        for col in combinedDf:
            plt.subplots(figsize = (10,5))
            sns.countplot(x='cluster_predicted',hue=col, data = combinedDf)
            st.pyplot()
            st.write('')
            st.write('')    

    
