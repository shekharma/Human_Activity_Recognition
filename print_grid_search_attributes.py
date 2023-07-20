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
