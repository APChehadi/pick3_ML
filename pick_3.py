import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib
import pickle
from collections import Counter



csv_filename = "pick_3_results.csv"
model_filename = "pick_3_predictor" 


def train_lotto(num_var):
    lotto_csv = pd.read_csv(csv_filename, names=["year", "month", "day", "midday_evening", "num_1", "num_2", "num_3"])
    lotto_csv = lotto_csv.dropna()
    
    X = lotto_csv.drop(["num_1", "num_2", "num_3"], axis=1)
    y = lotto_csv[f"num_{num_var}"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

    # tree_model = tree.DecisionTreeRegressor()
    tree_model = tree.ExtraTreeRegressor()
    tree_model.fit(X_train, y_train)

    with open(model_filename, "wb") as model_file:
        joblib.dump(tree_model, model_file)
        print(f"Done Training num_{num_var}")

    with open(model_filename, "rb") as model_file:
        tree_model = joblib.load(model_filename)
        result = tree_model.score(X_test, y_test)
        print(result)


def predict_lotto_num():
    year = input("year: ")
    month = input("month: ")
    day = input("day: ")
    midday_evening = input("[1/2] midday / evening: ")
    parameters = [[int(year), int(month), int(day), int(midday_evening)]]
    print("\nTraining...")

    num_choice = 1
    hopefully_winning_nums = []
    while num_choice < 4:
        train_lotto(str(num_choice))
        with open(model_filename, "rb") as model_file:
            tree_model = joblib.load(model_filename)
            predicted_num = int(tree_model.predict(parameters)[0])
            hopefully_winning_nums.append(predicted_num)
        num_choice += 1

    print(f"\n{hopefully_winning_nums}")



predict_lotto_num()