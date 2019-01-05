# @thal @gkara @nalif: Warning! If you try to run me without activating the venv, you deserve the 5 seconds of panic and dread that will follow. (>^-^)>

# For dataset processing
import pandas as pd

# For system functions
import sys

# Constants
DEBUG = True
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

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_train['Departure'])
df_train['Departure'] = le.transform(df_train['Departure'])
df_train['Arrival'] = le.transform(df_train['Arrival'])
df_test['Departure'] = le.transform(df_test['Departure'])
df_test['Arrival'] = le.transform(df_test['Arrival'])

import numpy as np

# Logistic Regression Approach
# from sklearn.linear_model import LogisticRegression

# Random Forest Approach
from sklearn.ensemble import RandomForestClassifier

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

# Logistic Regression Approach
# clf = LogisticRegression()

# Random Forest Approach
clf = RandomForestClassifier(n_jobs=2, random_state=0, max_features=5)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

import csv
with open(RES_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])




if(input("Do you want to sumbit results now?") == "y"):
    msg = input("Message: ")
    submit_result(RES_FILE, msg)
