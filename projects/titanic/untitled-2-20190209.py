
#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

from hpsklearn import HyperoptEstimator, pca, any_classifier, any_preprocessing, random_forest, extra_trees, gradient_boosting
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, label_binarize, OneHotEncoder, LabelEncoder
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, feature_selection, metrics
from pystacknet.pystacknet import StackNetClassifier
from xgboost import XGBClassifier
from hyperopt import tpe
import numpy as np

def main():
    
    # Download the data and split into training and test sets

    iris = load_iris()
    
    X = iris.data
    y = iris.target
    
    test_size = int(0.2 * len(y))
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-test_size]]
    y_train = y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_test = y[indices[-test_size:]]
    
    # for other datas, there will more complex data clearning
    
    
    
    # list all machine learning algorithms for hyper params tuning
    MLA = {
        'rfc':  [
                RandomForestClassifier(),
                #RandomForestClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
                {
                'n_estimators': [50,100,200], #default=1.0
                'criterion': ['entropy'], #edfault: auto
                'max_depth': [4,5,6], #default:ovr
                #'min_samples_split': [5,10,.03,.05,.10],
                'max_features': [.5],
                'random_state': [1],
                },
                random_forest('my_rfc'),
                ],
        
        'etc':  [
                ExtraTreesClassifier(), 
                #ExtraTreesClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
                {
                'n_estimators': [50,100,200], #default=1.0
                'criterion': ['entropy'], #edfault: auto
                'max_depth': [4,5,6], #default:ovr
                'max_features': [.5],
                'random_state': [1],
                },
                extra_trees('my_etc'),
                ],
        
        'gbc':  [
                GradientBoostingClassifier(),
                #GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
                {
                #'loss': ['deviance', 'exponential'],
                'learning_rate': [.1,.25,.5],
                'n_estimators': [50,100,200],
                #'criterion': ['friedman_mse', 'mse', 'mae'],
                'max_depth': [4,5,6],
                'max_features': [.5],
                #'min_samples_split': [5,10,.03,.05,.10],
                #'min_samples_leaf': [5,10,.03,.05,.10],      
                'random_state': [1],
                },
                gradient_boosting('my_rgc'),
                ], 
        
        'lr':  [
                LogisticRegression(),
                #LogisticRegression(random_state=1)
                {
                #'fit_intercept': grid_bool,
                #'penalty': ['l1','l2'],
                #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'random_state': [1],
                },
                ], 
        
        'svc':  [
                svm.SVC(),
                {
                #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default=1.0
                'gamma': [.1, .25, .5, .75, 1.0], #edfault: auto
                'decision_function_shape': ['ovo', 'ovr'], #default:ovr
                'probability': [True],
                'random_state': [0]
                },
                ],
    
        'xgb':  [
                XGBClassifier(),
                {
                #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
                'learning_rate': [.01, .03, .05, .1, .25], #default: .3
                'max_depth': [1,2,4,6,8,10], #default 2
                'n_estimators': [10, 50, 100, 300], 
                'seed': [0]  
                },
                ]    
        }

    # list some algorithms for HyperoptEstimator, but error !!!
    #MLA2 = {
        #'rfc':  [
                #random_forest('my_rfc'),
                #],
        
        #'etc':  [
                #extra_trees('my_etc'),
                #],
        
        #'gbc':  [
                #gradient_boosting('my_rgc'),
                #], 
 
        #}  
    # list some algorithms for HyperoptEstimator, but error !!!
    
    
    def opt(clf):
        est = MLA[clf][0]

        # ---------want to use Hyperopt, but has some errors !!!
        #estim = HyperoptEstimator(classifier=MLA2[clf][0],
                                  #preprocessing=[],
                                  #algo=tpe.suggest,
                                  #max_evals=3,
                                  #trial_timeout=120)
        
        #estim.fit( X_train, y_train )
        
        #est = estim
        
        # ---------want to use Hyperopt, but has some errors !!!
        
        # use GridSearchCV, it's too slow
        est = model_selection.GridSearchCV(estimator=est, param_grid=MLA[clf][1], cv=5) # --, scoring='roc_auc'
        
        return est
        
    # for StackNetClassifier
    #models=[ 
            ######### First level ########
            #[RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
             #ExtraTreesClassifier(n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
             #GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
             #LogisticRegression(random_state=1)
            #],
            ######### Second level ########
            #[RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)]
            #]
    
    models=[ 
            ######## First level ########
            [
            opt('rfc'),
            opt('etc'),
            #opt('gbc'),
            #opt('lr'),
            ],
            ######## Second level ########
            [
            opt('rfc'),
            ],
           ]
    
    # use StackNet to stacking the models
    StackNetmodel=StackNetClassifier(models, folds=4, # --metric="auc", 
                                     restacking=False, use_retraining=True, use_proba=True, 
                                     random_state=12345, n_jobs=1, verbose=1)
    
    StackNetmodel.fit(X_train, y_train)    
    
    

if __name__ == '__main__':
    print('Begin!')
    main()
    print('Finished!')