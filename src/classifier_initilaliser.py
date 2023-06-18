import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from skopt.space import Real, Integer
import numpy as np
from typing import Dict

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

class ClassifierInitialiser:
    """
    Initialises and trains the XGBoost classifier
    """

    def __init__(self) -> None:
        self.optmiser = None
        self.bayes_cv_tuner = None

    def tune_hyper_parameter(self, X : pd.DataFrame, y : pd.DataFrame, weights : pd.DataFrame, options : Dict) -> None:
        """
        Fits training data to the bayes cv hyper paramter tuning
        """
        try:
            self.model_options = options["CLASSIFIER_OPTIONS"]
            self.balance_scheme = options["BALANCE_SCHEME"]
            self.bayescv_options = options["BAYESCV_OPTIONS"]
            self.split_random_state = options["SPLIT_RANDOM_STATE"]
            self.test_size = options["TEST_SIZE"]
            self.n_jobs = self.bayescv_options["N_JOBS"]
            self.n_iter = self.bayescv_options["ITERATIONS"]
            self.scoring = self.bayescv_options["SCORING"]
            self.oversample_scheme = options["OVERSAMPLE_SCHEME"]
            self.undersample_scheme = options["UNDERSAMPLE_SCHEME"]
            self.is_compute_weights = options["IS_COMPUTE_WEIGHTS"]
            self.is_scale_pos_weight = options["IS_SCALE_POS_WEIGHT"]
            self.combinesample_scheme = options["COMBINESAMPLE_SCHEME"]
        except:
            # TODO: raise error
            pass
            
        # initialise the data
        self.X = X
        self.y = y
        self.weights = weights

        # Train test split
        if self.weights is not None:
            self.X_train, self.X_test, self.y_train, self.y_test , self.weights_train, self.weights_test = train_test_split(
                                                    X, y, weights,
                                                    test_size=self.test_size,
                                                    random_state=self.split_random_state,
                                                    shuffle=True,
                                                    stratify=y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                    X, y,
                                                    test_size=self.test_size,
                                                    random_state=self.split_random_state,
                                                    shuffle=True,
                                                    stratify=y)
            self.weights_train = None

        '''
        Check whether the task is multi-class or not. This is done to calculate
        the scale_pos_weight in XGBClassifier. It is also used for the under sampling
        and over sampling in the case for binary classification. 
        '''
        if self.is_scale_pos_weight and (len(self.y.iloc[:, 0].unique()) <= 2):
            minor_major_ratio = self.y_train.iloc[:, 0].sum() / (self.y_train.iloc[:, 0].size - self.y_train.iloc[:, 0].sum())
            oversampler_applier = OversampleApplier(self.oversample_scheme, minor_major_ratio)
            undersampler_applier = UndersampleApplier(self.undersample_scheme, minor_major_ratio)
            combinesampler_applier = CombinationSamplerApplier(self.combinesample_scheme, minor_major_ratio)
            xgb_applier = XgboostApplier(self.model_options, minor_major_ratio)
        else:
            oversampler_applier = OversampleApplier(self.oversample_scheme)
            undersampler_applier = UndersampleApplier(self.undersample_scheme)
            combinesampler_applier = CombinationSamplerApplier(self.combinesample_scheme)
            xgb_applier = XgboostApplier(self.model_options)

        # initialise the pipeline
        pipeline_list = [('oversampler', oversampler_applier.oversampler), 
                        ('undersampler', undersampler_applier.undersampler), 
                        ('combinesampler', combinesampler_applier.combinesampler), 
                        ('xgb', xgb_applier.model)]
        pipeline_list = [(k, v) for k, v in pipeline_list if v is not None]
        
        
        # If no over and under sampler is selected, return the xgb classifier as the pipeline
        self.fit_params = None
        self.needs_weights_computing = False
        if self.is_compute_weights:
            self.needs_weights_computing = True
            self.weights_train = compute_sample_weight(class_weight='balanced',y=self.y_train)
            self.fit_params = {'xgb__sample_weight' : self.weights_train}
        elif self.weights is not None:
            self.fit_params = {'xgb__sample_weight' : self.weights_train}

        self.pipeline = Pipeline(steps = pipeline_list)
        self.search_space = oversampler_applier.search_space | undersampler_applier.search_space | xgb_applier.search_space


        '''
        Initialising the hyperparameter tuning. A pipeline is tuned.
        The pipeline consists of over and under sampling depending on
        configuration. Or neither. 
        '''
        print("Fit params: ", self.fit_params)
        self.bayes_cv_tuner = BayesSearchCV(
            estimator = self.pipeline,
            search_spaces = self.search_space,
            scoring = self.scoring,
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=np.random.RandomState(0)
            ),
            n_jobs = self.n_jobs,
            n_iter = self.n_iter,
            verbose = 2,
            refit = True,
            random_state=np.random.RandomState(42),
            fit_params=self.fit_params
        )

        '''
        Where the fit actually occurs
        '''
        self.bayes_cv_tuner.fit(self.X_train, self.y_train)


    def get_prediction_df(self) -> pd.DataFrame:
        """
        Returns a dafaframe with the first column being the actual predictions
        and the following columns the probability for being each of the classes
        """
        if self.bayes_cv_tuner == None:
            # TODO: Error message
            return

        y_pred_df = pd.DataFrame(self.bayes_cv_tuner.predict(self.X_test))
        y_proba_df = pd.DataFrame(self.bayes_cv_tuner.predict_proba(self.X_test))

        return pd.concat([y_pred_df, y_proba_df], axis=1)
    


    def get_final_classifier(self):
        """
        Fits the entire data to the found hyper-paramters
        """
        if self.bayes_cv_tuner == None:
            # TODO: Error message
            return

        if self.is_compute_weights:
            self.weights = compute_sample_weight(
                                            class_weight='balanced',
                                            y=self.y)

        return self.bayes_cv_tuner.best_estimator_.fit(
                                        self.X, self.y,
                                        xgb__sample_weight = self.weights)



class XgboostApplier:
    """
    Itialises the XGBoost classifier.
    """
    def __init__(self, args : Dict, ratio : int=-1) -> None:
        try:
            self.model = xgb.XGBClassifier(
                    n_jobs = args["N_JOBS_MODEL"],
                    objective = args["OBJECTIVE_FUNC"],
                    tree_method='hist',
                    eval_metric = args["EVAL_METRIC"]
                )

            self.search_space = {
                        'xgb__eta' : Real(1e-8, 1, 'log-uniform'),
                        'xgb__reg_alpha' : Real(1e-8, 1.0, 'log-uniform'),
                        'xgb__reg_lambda': Real(1e-8, 1000, 'log-uniform'),
                        'xgb__max_depth': Integer(0, 50, 'uniform'),
                        'xgb__n_estimators': Integer(10, 300, 'uniform'),
                        'xgb__learning_rate': Real(1e-8, 1.0, 'log-uniform'),
                        'xgb__min_child_weight': Integer(0, 10, 'uniform'),
                        'xgb__max_delta_step': Integer(1, 100, 'uniform'),
                        'xgb__subsample': Real(1e-8, 1.0, 'uniform'),
                        'xgb__colsample_bytree': Real(1e-8, 1.0, 'uniform'),
                        'xgb__colsample_bylevel': (1e-8, 1.0, 'uniform'),
                        'xgb__gamma': Real(1e-8, 1.0, 'log-uniform'),
                        'xgb__min_child_weight': Integer(0, 5, 'uniform')
                    }
            if ratio > 0:
                self.search_space = self.search_space | {'xgb__scale_pos_weight': Real(1, 1/ratio)}
        except:
            # TODO: Raise an error
            pass



class OversampleApplier:
    """
    Over sample applier. Can choose from SMOTE and ADASYN.
    """
    def __init__(self,  oversampler_str : str, ratio : int=-1, random_state : int =42) -> None:
        self.valid_scheme = {
                            "SMOTE" : SMOTE(random_state=random_state),
                            "ADASYN" : ADASYN(random_state=random_state)
                            }
        
        self.search_space = {}
        if oversampler_str in self.valid_scheme:
            self.oversampler = self.valid_scheme[oversampler_str]
            
            if ratio > 0:
                self.search_space = {
                    'oversampler__sampling_strategy' : Real(ratio, 1, 'uniform')
                }
        else:
            self.oversampler = None
            print("Invalid oversampling scheme. Set to None")
        



class UndersampleApplier:
    """
    Under sample applier. Can choose from Tomek, ENN, Random Under Sampling
    """
    def __init__(self,  undersampler_str : str, ratio : int=-1, random_state : int =42) -> None:
        self.valid_scheme = {
                            "TOMEK" : TomekLinks(),
                            "ENN" : EditedNearestNeighbours(),
                            "RUS" : RandomUnderSampler(random_state=random_state)
                            }
        
        self.search_space = {}
        if undersampler_str in self.valid_scheme:
            self.undersampler = self.valid_scheme[undersampler_str]    
            if ratio > 0:
                self.search_space = {
                    'undersampler__sampling_strategy' : Real(ratio, 1, 'uniform')
                }
        else:
            self.undersampler = None
            print("Invalid undersampling scheme. Set to None")

class CombinationSamplerApplier:
    """
    Under sample applier. Can choose from Tomek, ENN, Random Under Sampling
    """
    def __init__(self,  combinesampler_str : str, ratio : int=-1, random_state : int =42) -> None:
        self.valid_scheme = {
                            "SMOTETOMEK" : SMOTETomek(random_state=random_state),
                            "SMOTEENN" : SMOTEENN(random_state=random_state)
                            }
        
        self.search_space = {}
        if combinesampler_str in self.valid_scheme:
            self.combinesampler = self.valid_scheme[combinesampler_str]    
            if ratio > 0:
                self.search_space = {
                    'combinesampler__sampling_strategy' : Real(ratio, 1, 'uniform')
                }
        else:
            self.combinesampler = None
            print("Invalid combinesampler scheme. Set to None")
        
