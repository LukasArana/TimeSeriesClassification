# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import time
from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.datasets import load_classification
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.convolution_based import RocketClassifier
from collections import Counter

NN_Euclidean = KNeighborsTimeSeriesClassifier(distance="euclidean", n_neighbors=1)
NN_DTW = KNeighborsTimeSeriesClassifier(distance="dtw", n_neighbors=1)
ST = ShapeletTransformClassifier()
Rocket = RocketClassifier()
tsf = TimeSeriesForestClassifier()
catch22 = Catch22Classifier()
be = BOSSEnsemble()

# Define classifiers
classifiers = {
    "NN_Euclidean": NN_Euclidean,
    "NN_DTW": NN_DTW,
    "ST": ST,
    "Rocket": Rocket,
    "Tsf": tsf,
    "Catch22": catch22,
    "Be": be
}

datasets = ["FreezerRegularTrain", "FordA", "ElectricDeviceDetection", "SharePriceIncrease", "Yoga", "PhalangesOutlinesCorrect", "ECGFiveDays",
            "Wafer", "RightWhaleCalls", "FordB", "SonyAIBORobotSurface2", "FreezerSmallTrain",
            "HandOutlines", "SemgHandGenderCh2", "Epilepsy2", "ItalyPowerDemand", "MoteStrain", "Strawberry", "TwoLeadECG"]
#Create smaller names for the datasets

def split(X, y):
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[y == 0], y[y == 0], test_size=100, random_state=42)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[y == 1], y[y == 1], test_size=100, random_state=42)

    X_train = np.concatenate((X_train_0, X_train_1), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1), axis=0)
    X_test = np.concatenate((X_test_0, X_test_1), axis=0)
    y_test = np.concatenate((y_test_0, y_test_1), axis=0)

    shuffle_index_train = np.random.permutation(len(X_train))
    shuffle_index_test = np.random.permutation(len(X_test))
    X_train, y_train = X_train[shuffle_index_train], y_train[shuffle_index_train]
    X_test, y_test = X_test[shuffle_index_test], y_test[shuffle_index_test]

    X_train = np.concatenate((X_train[y_train == 0][:250], X_train[y_train == 1][:250]), axis=0)
    y_train = np.concatenate((y_train[y_train == 0][:250], y_train[y_train == 1][:250]), axis=0)
    X_test = np.concatenate((X_test[y_test == 0][:150], X_test[y_test == 1][:150]), axis=0)
    y_test = np.concatenate((y_test[y_test == 0][:150], y_test[y_test == 1][:150]), axis=0)

    return X_train, y_train, X_test, y_test



def main():
    results = {dataset: {} for dataset in datasets}
    rs = np.random.RandomState(42)
    le = LabelEncoder()
    time_pd = {dataset: {classifier: None for classifier in classifiers} for dataset in datasets}
    # Iterate over datasets and models
    for dataset in datasets:
        print(dataset)
        X, y = load_classification(dataset)
        y = le.fit_transform(y)
        X = np.array(X)
        X_train, y_train, X_test, y_test = split(X,y)
        assert(X_train.shape == (500, 1, X_train.shape[2])) #Train is 500 and univariate
        assert(X_test.shape == (200, 1, X_test.shape[2])) # Test is 200 and univariate
        for clf_name, clf_model in classifiers.items():
            results = {}
            st = time.process_time()
            clf_model.fit(X_train, y_train)
            end = time.process_time()
            time_pd[dataset][clf_name] = end - st
            y_pred = clf_model.predict(X_test)
            results = {"pred": y_pred, "true": y_test}
            acc = accuracy_score(y_test, y_pred)
            pd.DataFrame(results).to_csv(f"results/{dataset}_{clf_name}.csv", index = False)
            pd.DataFrame(time_pd).to_csv(f"results/time.csv")
            print(f"{dataset}_{clf_name}")
            print("-------------------------------------------")
        print("********************************************")
if __name__ == "__main__":
    main()