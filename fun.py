import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family']='DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=90)
    plt.yticks(tick_marks,classes)
    
    fmt='.2f' if normalize else'd'
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j],fmt),
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True lable')
    plt.xlabel('Predicted label')




from datetime import datetime
def perform_model(model,x_train,y_train,x_test,y_test,class_labels,cm_normalize=True,\
                 print_cm=True, sm_cmap=plt.cm.Greens):
    
    #to store the results
    results=dict()
    
    #tiime at which model statrts training
    train_start_time=datetime.now()
    print('training the model...')
    model.fit(x_train,y_train)
    print('Done \n \n')
    train_end_time=datetime.now()
    results['training_time']=train_end_time-train_start_time
    print('training_time(HH:MM:SS.ms)- {}\n\n'.format(results['training_time']))
    
    #predict test datat
    print('Predicting test data')
    test_start_time=datetime.now()
    y_pred=model.predict(x_test)
    test_end_time=datetime.now()
    results['testing_time']=test_end_time-test_start_time
    print('testing_time(HH:MM:SS.ms)- {}\n\n'.format(results['testing_time']))
    results['predicted']=y_pred
    
    #calculate overall accuracy of the model 
    accuracy=metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    
    #stop accuracy in results
    results['accuracy']=accuracy
    print('-------------------------------------')
    print('|             Accuracy              |')
    print('-------------------------------------')
    print('\n    {}\n\n'.format(accuracy))
    
    # confusion matrix
    cm=metrics.confusion_matrix(y_test,y_pred)
    results['confusion_matrix']=cm
    if print_cm:
        print('-------------------------------------')
        print('|       Confusion Matrix             |')
        print('-------------------------------------')
        print('\n    {}'.format(cm))
        
    #plot confusion matrix
    plt.figure(figsize=(8,8))
    plt.grid(False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True,title="Normalized confusion matrix", cmap=sm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------------------')
    print('|   Classification report           |')
    print('-------------------------------------')
    classification_report=metrics.classification_report(y_test,y_pred)
    
    #store report in results
    results['classification_report']=classification_report
    print(classification_report)
    
    #add the trained model to the results
    results['model']=model
    return results



def print_grid_search_attributes(model):
    # Estimator that gave the highest score among all the estimators formed in Gridsearch
    print('---------------------------------')
    print('|           Best Estimator       |')
    print('----------------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))
    
    
    # parameters that gave best results while performing grid search
    print('---------------------------------')
    print('|           Best parameters       |')
    print('----------------------------------')
    print('\tParameters of the best estimator : \n\nt{}\n'.format(model.best_params_))
    
    
    # number of cross validation splits
    print('---------------------------------')
    print('|    No of cross validation sets      |')
    print('----------------------------------')
    print('\n\tTotal Numbers of cross validation sets:{}\n'.format(model.n_splits_))
    
    
    # Average cross validation score of the best estimator, from the grid search
    print('---------------------------------')
    print('|           Best score       |')
    print('----------------------------------')
    print('\n\tAverage cross validate scores of best estimator:\n\n\t{}\n'.format(model.best_score_))
