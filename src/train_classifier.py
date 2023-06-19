"""
Author: YOKOTE Kenta
Aim: To run the XGBoost classifier on labelled cell data on SLURM
    Takes 8 inputs from STDIN:
        1. run_name: name of the run. The outputs will be saved to a folder
                     which has this as the name
        2. input_folder: 
        3. input_file: name of the file located in input_folder
        4. labels_file: name of the file containing labels 
        5. output_folder: 
        6. classifier_scheme: the type of classifier to use
        7. model_options: 
"""

import sys
import json
import pandas as pd
import pickle
from classifier_initilaliser import ClassifierInitialiser
from preprocess.data_transformer import DataTransformer
import os

if __name__ == '__main__':
    # Run options
    run_options_file = sys.argv[1]
    with open(run_options_file) as json_file:
        run_options = json.load(json_file) 

    # Run name
    run_name = run_options["RUN_NAME"]
    
    # Get the input variables 
    input_file = run_options["INPUT_FILE"]
    # Get the labels file
    labels_file = run_options["LABELS_FILE"]

    # Get the output folder
    output_folder = run_options["OUTPUT_FOLDER"]

    # Preprocess
    preprocess_scheme = run_options["PREPROCESS_SCHEME"]
    preprocess_options = run_options["PREPROCESS_OPTIONS"]

    # model options
    model_options = run_options["MODEL_OPTIONS"]

    # classifier
    classifier_scheme = run_options["CLASSIFIER"]

    # Save preprocessed data
    try:
        save_preprocessed_data = run_options["SAVE_PREPROCESSED"]
    except:
        save_preprocessed_data = False

    # Make output directory if not already made 
    full_output_directory = output_folder + run_name + "/"
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)

    # Read the data
    print("INFO: Loading data")
    print("INFO: Measurement file: {}".format(input_file))
    X = pd.read_csv(input_file)
    y = pd.read_csv(labels_file)

    # Preprocess
    print("INFO: Preprocessing")
    data_transformer = DataTransformer()
    X = data_transformer.transform_data(X, 
                                transform_scheme=preprocess_scheme, 
                                args=preprocess_options)
    if save_preprocessed_data:
        X.to_csv(full_output_directory + "preprocessed_data.csv", index=False)

    # Tune the hyperparameters 
    print("INFO: Tuning hyperparameters")
    classifier_applier = ClassifierInitialiser()
    classifier_applier.tune_hyper_parameter(X, y, classifier_scheme, model_options) 

    # get predictions
    print("INFO: Save the predictions")
    classifier_applier.get_prediction_df().to_csv(full_output_directory + "y_predicted.csv", index=False)
    classifier_applier.y_train_orig.to_csv(full_output_directory + "y_train.csv", index=False)
    classifier_applier.y_test.to_csv(full_output_directory + "y_test.csv", index=False)

    # Save the outputs 
    print("INFO: Save the cross validation results")
    filename_bayes = full_output_directory + "bayes_cv_model.sav"
    pickle.dump(classifier_applier.bayes_cv_tuner, open(filename_bayes, 'wb'))

    # Sav ethe final best model
    print("INFO: Save final model")
    model = classifier_applier.get_final_classifier()
    filename = full_output_directory + "final_model.sav"
    pickle.dump(model, open(filename, 'wb'))
    print("INFO: Finished")