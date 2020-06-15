#!/bin/python

class Config:
    # Directories
    data_dir = 'Data'
    results_dir = 'Results'
    plot_dir = 'Plots'
    hyperparams_dir = 'Hyperparameters'
    shap_dir = 'SHAP'

    # Following admission
    time_window = 36  # hours

    # Training related
    random_state = 42
    values_present = 0.7  # At least this much should be present
    validation_folds = 10

    # Hyperparam related
    gs_iterations = 5000
    hyperparameter_pickle = 'Hyperparams'

    # Classifier descriptions
    # NOTE - Also decides which classifiers are going to be run
    clf_descriptions = {
        'XGB_Imputed': 'Hyperparams_xgb_imputed.pickle',
        'XGB_NotImputed': 'Hyperparams_xgb.pickle',
        'LogisticRegression': None,
        'LASSO': None,
    }

    # Database related
    database_path = '../COVID.db'
    cutoff_date = '2020/05/01'
