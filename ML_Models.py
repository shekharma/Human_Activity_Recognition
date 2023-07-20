# Import Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")    # Set the warning filter to ignore


######## Provide path of data
base_path='C:/Users/Administrator/Documents/HAR_Kaggle/'    ## Provide directory of downloaded folder
train=pd.read_csv(base_path + 'train.csv')   ## read the 'train.csv' file from HAR_Kaggle folder
test=pd.read_csv(base_path+ 'test.csv')      ## read the 'test.csv' file from HAR_Kaggle folder

# removing symbols from feature names
columns=train.columns

#removing '()','-',',' from columns by empty string('')
columns=columns.str.replace('[()]','')
columns=columns.str.replace('[-]','')
columns=columns.str.replace('[,]','')

#after removing this symbols we rename the columns
train.columns=columns
test.columns=columns

#preveiw of column names
test.columns

# get x_train, and y_train from csv files
x_train=train.drop(['subject','Activity'],axis=1)
y_train=train.Activity

#get x_test and y_test from csv files
x_test=test.drop(['subject','Activity'], axis=1)
y_test=test.Activity

print('x_train and y_train: ({},{})'.format(x_train.shape,y_train.shape))
print('x_test and y_test: ({},{})'.format(x_test.shape,y_test.shape))

             ##         ##
             ### MODEL ###
             ##         ##

## Labels for plottin confusion Matrix ##
labels=['LAYING','SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family']='DejaVu Sans'

################ FUNCTIONS ################
## Function for confusion Matrix ##
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    ## Import function from plot_confusion_matrix.py

                             
## function to run the model ##
from datetime import datetime
def perform_model(model,x_train,y_train,x_test,y_test,class_labels,cm_normalize=True,\
                 print_cm=True, sm_cmap=plt.cm.Greens):
    ## Import function from perfom_model.py

                     
## Function to print Gridsearch attributes ##
def print_grid_search_attributes(model):
     ## Import print_grid_search_atrributes.py

    
############################ 1. Logistic Regression With GridSearch ############################
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# start Grid search
parameters={'C':[0.01,0.1,1,10,20,30], 'penalty':['l2','l2']}   ##'c':[0.01,0.1,1,10,20,30]
log_reg=linear_model.LogisticRegression()
log_reg_grid=GridSearchCV(log_reg, param_grid=parameters,cv=3, verbose=1, n_jobs=-1, error_score='raise')
log_reg_grid_results=perform_model(log_reg_grid,x_train, y_train, x_test, y_test, class_labels=labels)

#### Plot Confusion Matrix ####
plt.figure(figsize=(8,8))
plt.grid(False)
plot_confusion_matrix(log_reg_grid_results['confusion_matrix'], classes =labels, cmap=plt.cm.Greens,)
plt.show()


############################ 2. Linear SVC with GrideSearch ############################
from sklearn.svm import LinearSVC
parameters={'C':[0.125,0.5,1,2,8,16]}
lr_svc=LinearSVC(tol=0.00005)
lr_svc_grid=GridSearchCV(lr_svc, param_grid=parameters, n_jobs=1, verbose=1)
lr_svc_grid_results= perform_model(lr_svc_grid, x_train, y_train, x_test,y_test, class_labels=labels)


############################ 3. Kernel SVM with Gridsearch ############################
from sklearn.svm import SVC 
parameters={'C':[2,8,16],\
           'gamma':[0.0078125,0.125,2]}
rbf_svm=SVC(kernel='rbf')
rbf_svm_grid=GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
rbf_svm_grid_results=perform_model(rbf_svm_grid, x_train, y_train, x_test, y_test, class_labels=labels)


############################ 4. Decision Trees with Gridsearch ############################
from sklearn.tree import DecisionTreeClassifier 
parameters={'max_depth':np.arange(3,10,2)}
dt= DecisionTreeClassifier()
dt_grid=GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
dt_grid_results=perform_model(dt_grid, x_train, y_train, x_test, y_test, class_labels=labels)
print_grid_search_attributes(dt_grid_results['model'])


############################ 5. Random forest Classifier with Gridsearch ############################
from sklearn.ensemble import RandomForestClassifier 
params={'n_estimators':np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
rfc= RandomForestClassifier()
rfc_grid=GridSearchCV(rfc,param_grid=params, n_jobs=-1)
rfc_grid_results=perform_model(rfc_grid, x_train, y_train, x_test, y_test, class_labels=labels)
print_grid_search_attributes(rfc_grid_results['model'])

############################ 6. Gradient Boosted decision trees with Gridsearch ############################
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier 
param_grid={'n_estimators':np.arange(130,170,10), 'max_depth':np.arange(5,8,1)}
gbdt= GradientBoostingClassifier()
gbdt_grid=GridSearchCV(gbdt,param_grid=param_grid, n_jobs=-1)
gbdt_grid_results=perform_model(gbdt_grid, x_train, y_train, x_test, y_test, class_labels=labels)
print_grid_search_attributes(gbdt_grid_results['model'])

############ Comparing All Models ############
print('\n                   Accuracy     Error')
print('                     --------     ------')
print('Logistic Regression :{:.04}%      {:.04}%'.format(log_reg_grid_results['accuracy']*100,\
                                                        100-(log_reg_grid_results['accuracy']*100)))

print('Linear SVC          :{:.04}%      {:.04}%'.format(lr_svc_grid_results['accuracy']*100,\
                                                        100-(lr_svc_grid_results['accuracy']*100)))

print('rbf SVM classifier  :{:.04}%      {:.04}%'.format(rbf_svm_grid_results['accuracy']*100,\
                                                        100-(rbf_svm_grid_results['accuracy']*100)))

print('Decision Tree       :{:.04}%      {:.04}%'.format(dt_grid_results['accuracy']*100,\
                                                        100-(dt_grid_results['accuracy']*100)))

print('Random Forest       :{:.04}%      {:.04}%'.format(rfc_grid_results['accuracy']*100,\
                                                        100-(rfc_grid_results['accuracy']*100)))

print('GradientBoosting DT :{:.04}%      {:.04}%'.format(gbdt_grid_results['accuracy']*100,\
                                                        100-(gbdt_grid_results['accuracy']*100)))
