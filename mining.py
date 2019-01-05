# @thal @gkara @nalif: Warning! If you try to run me without activating the venv, you deserve the 5 seconds of panic and dread that will follow. (>^-^)>

# For dataset processing
import pandas as pd
import numpy as np

# For system functions
import sys

# For comparison between models and optimization
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Constants
DEBUG = True
THREADS = 10
COMPETITION = "inf131-data-mining"
RES_FILE = "../y_pred.csv"
TRAINING_DATA = "../train.csv"
TEST_DATA = "../test.csv"

# Helper Functions
def errlog(error):
    print("")
    print("======ERROR START======")
    print("Halting and catching fire!")
    print("Potential cause: " + error)
    print("=======ERROR END=======")
    print("")
    sys.exit(1)

def dbglog(info, target):
    if DEBUG:
        print("")
        print("======DEBUG BLOCK======")
        print("Debug Info: " + target)
        print(info)
        print("=======DEBUG END=======")
        print("")

def submit_result(data, message):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except:
        errlog("Kaggle API not found. Maybe install it via pip and make sure it is in the path, or venv, or whatever weird solution you are using to defeat pip's stupidity?")
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(data, message, COMPETITION)

# Algorithm Comparison
def compare_algos(X, Y):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(max_depth=32, min_samples_split=40)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma="scale")))
    models.append(('RF', RandomForestClassifier(n_jobs=THREADS, random_state=0, max_features=5)))

    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "Classification Comparison")    

# Max Depth Comparison
def max_depth_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    max_depths = np.linspace(1, 64, 64, endpoint=True)

    for max_depth in max_depths:
        models.append(('RF_' + str(max_depth), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_features=5, max_depth=max_depth)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "max_depth comparison")

# Estimator Count Comparison
def n_estimators_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

    for n_estimator in n_estimators:
        models.append(('RF_' + str(n_estimator), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_features=5, max_depth=13, n_estimators=n_estimator)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "n_estimators comparison")

# Samples Split Comparison
def min_samples_split_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    #min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) -- Bad results when ceil'ed by the RF implementation of SciKit.
    min_samples_splits = [2,3,4,5,6,7,8,9,10,11]

    for min_samples_split in min_samples_splits:
        models.append(('RF_' + str(min_samples_split), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_features=5, max_depth=13, n_estimators=64, min_samples_split=min_samples_split)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "min_samples_split comparison")

# Samples Leaf Comparison
def min_samples_leaf_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    min_samples_leaves = [2,3,4,5,6,7,8,9,10,11]

    for min_samples_leaf in min_samples_leaves:
        models.append(('RF_' + str(min_samples_leaf), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_features=5, max_depth=13, n_estimators=64, min_samples_split=6, min_samples_leaf=min_samples_leaf)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "min_samples_leaf comparison")

# Max Features Comparison
def max_features_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    max_features_numbers = [1,2,3,4,5,6, "auto", "sqrt", "log2"]

    for max_features_number in max_features_numbers:
        models.append(('RF_' + str(max_features_number), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_depth=13, n_estimators=64, min_samples_split=6, min_samples_leaf=2, max_features=max_features_number)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "max_features comparison")

# Criteria Comparison
def criteria_comparison(X, Y):
    models = []
    results = []
    names = []
    scoring = 'accuracy'
    msgs = ""
    criteria_choices = ["gini", "entropy"]

    for criteria_choice in criteria_choices:
        models.append(('RF_' + str(criteria_choice), RandomForestClassifier(n_jobs=THREADS, random_state=0, max_depth=13, n_estimators=64, min_samples_split=6, min_samples_leaf=2, max_features=5, criterion=criteria_choice)))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
        msgs += msg
    dbglog(msgs, "criteria comparison")

# Import training data
df_train = pd.read_csv(TRAINING_DATA)
dbglog(df_train.head(), "Initial Training Data")

# Import test data
df_test = pd.read_csv(TEST_DATA)
dbglog(df_test.head(), "Initial Test Data")

# Save training target before dropping
y_train = df_train[['PAX']]

# Calculate Week Day - Busy weekdays have more passengers than others.
import datetime

def dateToWeekday(row):
    return datetime.datetime.strptime(row["DateOfDeparture"], '%Y-%m-%d').weekday()

def isWeekend(row):
    if datetime.datetime.strptime(row["DateOfDeparture"], '%Y-%m-%d').weekday() in [5,6]:
        return 1
    return 0

import math

def long_lat_to_x(row, isDep=False):
    if isDep:
        return math.cos(row["LatitudeDeparture"]) * math.cos(row["LongitudeDeparture"])
    return math.cos(row["LatitudeArrival"]) * math.cos(row["LongitudeArrival"])

def long_lat_to_y(row, isDep=False):
    if isDep:
        return math.cos(row["LatitudeDeparture"]) * math.sin(row["LongitudeDeparture"])
    return math.cos(row["LatitudeArrival"]) * math.sin(row["LongitudeArrival"])

def long_lat_to_z(row, isDep=False):
    if isDep:
        return math.sin(row["LatitudeDeparture"])
    return math.sin(row["LatitudeArrival"]) 

def tripDistance(row):
    p1 = np.array((row["DepX"], row["DepY"], row["DepZ"]))
    p2 = np.array((row["ArX"], row["ArX"], row["ArX"]))

    squared_dist = np.sum(p1**2 + p2**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

# Calculate Holidays based on USFederalHolidayCalendar
# from pandas.tseries.holiday import USFederalHolidayCalendar
# cal = USFederalHolidayCalendar()
# holidays = cal.holidays(start='1999-01-01', end='2018-12-31').to_pydatetime()

# Christmas - 23rd of December + 7 days
christmas = datetime.datetime.strptime('12-23','%m-%d')
christmas_week_dates = [ christmas + datetime.timedelta(days=i) for i in range(10) ]

# Summer Vacation - Whole August
summer_vacation = datetime.datetime.strptime('8-1','%m-%d')
summer_vacation_week_dates = [ summer_vacation + datetime.timedelta(days=i) for i in range(31) ]

# Fourth of July - 3 July - 5 July - Sudden rise in travelling
fourth_of_july = datetime.datetime.strptime('7-3','%m-%d')
fourth_of_july_week_dates = [ summer_vacation + datetime.timedelta(days=i) for i in range(2) ]

# List of busiest US airports
busy_airports = [
    "JFK",
    "LAX",
    "MIA",
    "SFO",
    "ORD",
    "EWR",
    "ATL",
    "IAH",
    "DFW",
    "IAD",
]

def isArrivalBusy(row):
    if row["Arrival"] in busy_airports:
        return 1
    return 0

def isDepartureBusy(row):
    if row["Departure"] in busy_airports:
        return 1
    return 0

def isHoliday(row):
    # Calculate "hot" holidays for traveling seperately from generic holidays.
    dt = datetime.datetime.strptime(row["DateOfDeparture"], '%Y-%m-%d')

    #for fourth_of_july_dt in fourth_of_july_week_dates:
    #    if dt == fourth_of_july_dt.replace(year=dt.year):
    #        return 3
    for christmas_dt in christmas_week_dates:
        if dt == christmas_dt.replace(year=dt.year):
            return 2
    for summer_vacation_dt in summer_vacation_week_dates:
        if dt == summer_vacation_dt.replace(year=dt.year):
            return 1
    return 0 

# Apply Weekday
df_train['Weekday'] = df_train.apply (lambda row: dateToWeekday (row),axis=1)
df_test['Weekday'] = df_test.apply (lambda row: dateToWeekday (row),axis=1)

dbglog(df_train.head(), "Training Data - Weekday")

# Apply IsHoliday
df_train['Holiday'] = df_train.apply (lambda row: isHoliday (row),axis=1)
df_test['Holiday'] = df_test.apply (lambda row: isHoliday (row),axis=1)

dbglog(df_train.head(), "Training Data - Holiday")

# Apply IsWeekend
# df_train['Weekend'] = df_train.apply (lambda row: isWeekend (row),axis=1)
# df_test['Weekend'] = df_test.apply (lambda row: isWeekend (row),axis=1)

# dbglog(df_train.head(), "Training Data - Weekend")

# Apply IsArrivalBusy
df_train['ArrivalBusy'] = df_train.apply (lambda row: isArrivalBusy (row),axis=1)
df_test['ArrivalBusy'] = df_test.apply (lambda row: isArrivalBusy (row),axis=1)

dbglog(df_train.head(), "Training Data - ArrivalBusy")

# Apply IsDepartureBusy
# df_train['DepartureBusy'] = df_train.apply (lambda row: isDepartureBusy (row),axis=1)
# df_test['DepartureBusy'] = df_test.apply (lambda row: isDepartureBusy (row),axis=1)

# dbglog(df_train.head(), "Training Data - DepartureBusy")

# Apply LongLat To DepXYZ
df_train['DepX'] = df_train.apply (lambda row: long_lat_to_x (row, isDep=True),axis=1)
df_train['DepY'] = df_train.apply (lambda row: long_lat_to_y (row, isDep=True),axis=1)
df_train['DepZ'] = df_train.apply (lambda row: long_lat_to_z (row, isDep=True),axis=1)

df_test['DepX'] = df_test.apply (lambda row: long_lat_to_x (row, isDep=True),axis=1)
df_test['DepY'] = df_test.apply (lambda row: long_lat_to_y (row, isDep=True),axis=1)
df_test['DepZ'] = df_test.apply (lambda row: long_lat_to_z (row, isDep=True),axis=1)

# Apply LongLat To ArXYZ
df_train['ArX'] = df_train.apply (lambda row: long_lat_to_x (row),axis=1)
df_train['ArY'] = df_train.apply (lambda row: long_lat_to_y (row),axis=1)
df_train['ArZ'] = df_train.apply (lambda row: long_lat_to_z (row),axis=1)

df_test['ArX'] = df_test.apply (lambda row: long_lat_to_x (row),axis=1)
df_test['ArY'] = df_test.apply (lambda row: long_lat_to_y (row),axis=1)
df_test['ArZ'] = df_test.apply (lambda row: long_lat_to_z (row),axis=1)

dbglog(df_train.head(), "Training Data - XYZ")

# Apply Trip Distance
df_train['Distance'] = df_train.apply (lambda row: tripDistance(row),axis=1)
df_test['Distance'] = df_test.apply (lambda row: tripDistance(row),axis=1)

dbglog(df_train.head(), "Training Data - Distance")

# Prepare training and data set
# Drop unused columns
# 0: DateOfDeparture
# 1: Departure
# 2: CityDeparture
# 3: Longtitude
# 4: Latitude
# 5: Arrival
# 6: CityArrival
# 7: LongtitudeArrival
# 8: LatitudeArrival
# 9: WeeksToDeparture
# 10: std_wtd
# 11: PAX (Only in Training Data)

df_train.drop(df_train.columns[[0,2,3,4,6,7,8,9,10,11]], axis=1, inplace=True)
df_test.drop(df_test.columns[[0,2,3,4,6,7,8,9,10]], axis=1, inplace=True)

df_train.drop(["DepX", "DepY", "DepZ", "ArX", "ArY", "ArZ"], axis=1, inplace=True)
df_test.drop(["DepX", "DepY", "DepZ", "ArX", "ArY", "ArZ"], axis=1, inplace=True)

dbglog(df_train.head(), "Training Data - Post Drops")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_train['Departure'])
df_train['Departure'] = le.transform(df_train['Departure'])
df_train['Arrival'] = le.transform(df_train['Arrival'])
df_test['Departure'] = le.transform(df_test['Departure'])
df_test['Arrival'] = le.transform(df_test['Arrival'])

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

# No FutureWarning(s) please. We shall handle this in a better manner in the future. (Oh, the irony)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Algo comparison
# compare_algos(X_train, y_train) -- Battle between RFs and CART. Let's make RFs win :P 

# Let's optimize our RF and hope for the best! :D

# max_depth comparison
# max_depth_comparison(X_train, y_train)
# Winner is: 13

# n_estimators comparison
# n_estimators_comparison(X_train, y_train)
# Winner is: 64 

# min_samples_split comparison
# min_samples_split_comparison(X_train, y_train)
# Winner is: 6

# min_samples_leaf comparison
# min_samples_leaf_comparison(X_train, y_train)
# Winner is: 2

# max_features comparison
# max_features_comparison(X_train, y_train)
# Winner is: 5

# criteria comparison
# criteria_comparison(X_train, y_train)
# Winner is: gini

# Based on comparisons above
clf = RandomForestClassifier(n_jobs=THREADS, random_state=0, max_depth = 13, n_estimators=64, min_samples_split=6, min_samples_leaf=2, max_features=6, criterion="gini") 

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

import csv
with open(RES_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])

if(input("Do you want to sumbit results now? (y/N): ") == "y"):
    msg = input("Message: ")
    submit_result(RES_FILE, msg)
