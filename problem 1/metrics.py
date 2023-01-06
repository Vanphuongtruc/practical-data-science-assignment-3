import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc


def metrics(y_train,y_hat_train,y_test,y_hat_test):
    print(f'Training Precision: ', round(precision_score(y_train, y_hat_train),5))
    print(f'Testing Precision: ', round(precision_score(y_test, y_hat_test),5))
    print('\n')
    print(f'Training Recall: ',round(recall_score(y_train,y_hat_train),5))
    print(f'Testing Recall: ',round(recall_score(y_test, y_hat_test),5))
    print('\n')
    print(f'Training Accuracy: ',round(accuracy_score(y_train,y_hat_train),5))
    print(f'Testing Accuracy: ',round(accuracy_score(y_test, y_hat_test),5))
    print('\n')
    print(f'Training F1-score: ', round(f1_score(y_train, y_hat_train),5))
    print(f'Testing F1-score: ', round(f1_score(y_test, y_hat_test),5))
    
def print_metric_comparisons(X, y):
    
    # Create empty lists to store metrics
    train_pre = []
    test_pre = []
    train_recall = []
    test_recall = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []
    
    # Iterate through a range of test_sizes to use for our logistic regression, using same parameters as our first logistic regression in our notebook. Append each respective result metric to its respective list.
    for i in range(10, 95):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i/100.0, random_state = 33)
        logreg = LogisticRegression(fit_intercept=False, C=1e25, solver='liblinear')
        model_log = logreg.fit(X_train, y_train)
        y_hat_test = logreg.predict(X_test)
        y_hat_train = logreg.predict(X_train)

        train_pre.append(precision_score(y_train, y_hat_train))
        test_pre.append(precision_score(y_test, y_hat_test))
        train_recall.append(recall_score(y_train, y_hat_train))
        test_recall.append(recall_score(y_test, y_hat_test))
        train_acc.append(accuracy_score(y_train, y_hat_train))
        test_acc.append(accuracy_score(y_test, y_hat_test))
        train_f1.append(f1_score(y_train, y_hat_train))
        test_f1.append(f1_score(y_test, y_hat_test))
        
    # Use subplots to create a scatter plot of each of the 4 metrics. 
    plt.figure(figsize = (20, 10))
    plt.subplot(221)
    plt.title('Precision', fontweight = 'bold', fontsize = 30)
    # Scatter plot training precision list
    plt.scatter(list(range(10, 95)), train_pre, label='Train',color='#FD6787')
    # Scatte4r plot test precision list
    plt.scatter(list(range(10, 95)), test_pre, label='Test',color='#288EEB')
    plt.xlabel('Model Test Size (%)', fontsize = 20)
    plt.legend(loc = 'best')

    plt.subplot(222)
    plt.title('Recall', fontweight = 'bold', fontsize = 30)
    # Scatter plot training recall list
    plt.scatter(list(range(10, 95)), train_recall, label='Train',color='#FD6787')
    # Scatter plot test recall list
    plt.scatter(list(range(10, 95)), test_recall, label='Test',color='#288EEB')
    plt.xlabel('Model Test Size (%)', fontsize = 20)
    plt.legend(loc = 'best')

    plt.subplot(223)
    plt.title('Accuracy', fontweight = 'bold', fontsize = 30)
    # Scatter plot training accuracy list
    plt.scatter(list(range(10, 95)), train_acc, label='Train',color='#FD6787')
    # Scatter plot test accuracy list
    plt.scatter(list(range(10, 95)), test_acc, label='Test',color='#288EEB')
    plt.xlabel('Model Test Size (%)', fontsize = 20)
    plt.legend(loc = 'best')

    plt.subplot(224)
    plt.title('F1', fontweight = 'bold', fontsize = 30)
    # Scatter plot training f1-score list
    plt.scatter(list(range(10, 95)), train_f1, label='Trail',color='#FD6787')
    # Scatter plot testing f1-score list
    plt.scatter(list(range(10, 95)), test_f1, label='Train',color='#288EEB')
    plt.xlabel('Model Test Size (%)', fontsize = 20)
    plt.legend(loc = 'best')

    plt.tight_layout()


def print_auc(model, X_train, X_test, y_train, y_test):
        # calculate the probability score of train set
        y_train_score = model.decision_function(X_train)
        # calculate false positive rate, true positive rate, thresolds of train set
        train_fpr, train_tpr, train_thre = roc_curve(y_train,y_train_score)

        # calculate the probability score of test set
        y_test_score = model.decision_function(X_test)
        # calculate false positive rate, true positive rate, thresolds of train set
        test_fpr, test_tpr, test_thre = roc_curve(y_test,y_test_score)

        #print auc score
        print('Train AUC: {}'.format(auc(train_fpr,train_tpr)))
        print('Test AUC: {}'.format(auc(test_fpr,test_tpr)))
        
        plt.figure(figsize=(20,10))
        lw = 2

        # plot receiver operating characteristic curve using false/true positive ratios for train set
        plt.subplot(121)
        plt.plot(train_fpr,train_tpr,color = '#FD6787',lw=lw, label= 'ROC Curve')
        # plot positive line with slope = 1 for roc-curve reference
        plt.plot([0,1],[0,1],color='#288EEB',linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.yticks([i/20.0 for i in range(21)])
        plt.xticks([i/20.0 for i in range(21)])
        plt.xlabel('False Positive', fontsize = 25)
        plt.ylabel('True Positive', fontsize = 25)
        plt.title('ROC Curve for Train Set', fontweight = 'bold', fontsize = 30)
        plt.legend(loc='lower right')

        # plot receiver operating characteristic curve using false/true positive ratios for test set
        plt.subplot(122)
        plt.plot(test_fpr, test_tpr, color='#FD6787',
            lw=lw, label='ROC curve')
        # Plot positive line w/ slope = 1 for ROC-curve reference
        plt.plot([0, 1], [0, 1], color='#288EEB', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.yticks([i/20.0 for i in range(21)])
        plt.xticks([i/20.0 for i in range(21)])
        plt.xlabel('False Positive', fontsize = 25)
        plt.ylabel('True Positive', fontsize = 25)
        plt.title('ROC Curve for Test Set', fontweight = 'bold', fontsize = 30)
        plt.legend(loc='lower right')
        
        plt.tight_layout()

def best_k(X_train,X_test,y_train,y_test):
    best_k = 0
    best_score = 0.0
    min_k = 1
    max_k=100

    for k in range(min_k,max_k+1,2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        preds = knn.predict(X_test)
        f1=f1_score(y_test,preds)
        if f1>best_score:
            best_k = k
            best_score = f1
    
    print(f'Best K values:{best_k}')
    print(f'F1: {best_score}')

def viz_roc(fpr,tpr,roc,string):
    plt.style.use('ggplot')
    plt.figure(figsize=(20,10))
    plt.plot([0,1],[0,1],lw=2,ls='--')
    plt.plot(fpr,tpr,lw=2, label = f'{string} AUC = '+str(roc))
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False positive', fontsize = 25)
    plt.ylabel('True Positive', fontsize = 25)
    plt.title(f'ROC curve: {string}', fontsize=30, fontweight='bold')
    plt.legend(loc=4, fontsize=15)
    plt.tight_layout()



