import pandas as pd # to import csv and for data manipulation
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import warnings
import os

from sklearn.utils import resample

warnings.filterwarnings('ignore')

class prediction():


    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/Accident_Information.csv')
        df = pd.read_csv(filename)
        print("file loading done")

        features_to_use = ['1st_Road_Class',
                     'Carriageway_Hazards',
                     'Day_of_Week',
                     'Junction_Detail',
                     'Light_Conditions',
                     'Road_Surface_Conditions',
                     'Road_Type',
                     'Special_Conditions_at_Site',
                     'Speed_limit',
                     'Time',
                     'Urban_or_Rural_Area',
                     'Weather_Conditions']

        label = ['Accident_Severity']
        total_cols = features_to_use + label
        df = df[total_cols]

        self.features = features_to_use
        self.label = label
        self.df = df
        df_train, df_test = self.preprocess(df)
        print("preprocessing done")

        self.model = self.Classifier(df_train, df_test, smallier_subset=True)
        print("model fitting done")
        print("launching application")

    #drop data missing and data not found rows
    def update(self, df):
        self.df = df
        self.features = list(df.drop(columns=self.label).columns)
        print(self.features)

    def handle_Road_Type(self, df):
        df = df[df['Road_Type'] != 'Unknown']
        encoding = {
            "Road_Type":
                {"Single carriageway": 1,
                 "Dual carriageway": 2,
                 "Roundabout": 3,
                 "One way street": 4,
                 "Slip road": 5}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Carriageway_Hazards(self, df):
        df = df[df['Carriageway_Hazards'] != 'Data missing or out of range']
        encoding = {
            "Carriageway_Hazards": {"None": 0,
                                    "Other object on road": 1,
                                    "Any animal in carriageway (except ridden horse)": 1,
                                    "Pedestrian in carriageway - not injured": 1,
                                    "Previous accident": 1,
                                    "Vehicle load on road": 1}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Light_Condition(self, df):
        # deal with column 'Light_Conditions', filter missing data and  unknown lighting
        df = df[df['Light_Conditions'] != 'Data missing or out of range']
        df = df[df['Light_Conditions'] != 'Darkness - lighting unknown']
        encoding = {
            "Light_Conditions": {"Daylight": 0,
                                 "Darkness - lights lit": 1,
                                 "Darkness - no lighting": 2,
                                 "Darkness - lights unlit": 2}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_day_of_week(self, df):
        # coonvert catagories of column day_of_week into two categories: weekday and weekend
        encoding = {
            "Day_of_Week": {"Monday": 0,
                            "Tuesday": 0,
                            "Wednesday": 0,
                            "Thursday": 0,
                            "Friday": 0,
                            "Saturday": 1,
                            "Sunday": 1}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Special_Conditions_at_Site(self, df):
        df = df[df['Special_Conditions_at_Site'] != 'Data missing or out of range']
        encoding = {
            "Special_Conditions_at_Site":
                {"None": 0,
                 "Roadworks": 1,
                 "Oil or diesel": 2,
                 "Mud": 3,
                 "Road surface defective": 4,
                 "Auto traffic signal - out": 5,
                 "Road sign or marking defective or obscured": 6,
                 "Auto signal part defective": 7}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_1st_Road_Class(self, df):
        encoding = {
            "1st_Road_Class": {"A": 1,
                               "A(M)": 1,
                               "B": 2,
                               "C": 3,
                               "Motorway": 4,
                               "Unclassified": 5}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Junction_Detail(self, df):
        df = df[df['Junction_Detail'] != 'Data missing or out of range']
        encoding = {
            "Junction_Detail":
                {"Not at junction or within 20 metres": 1,
                 "T or staggered junction": 2,
                 "Crossroads": 3,
                 "Roundabout": 4,
                 "Private drive or entrance": 5,
                 "Other junction": 6,
                 "Slip road": 7,
                 "More than 4 arms (not roundabout)": 8,
                 "Mini-roundabout": 9}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Road_Surface_Type(self, df):
        df = df[df["Road_Surface_Conditions"] != 'Data missing or out of range']
        encoding = {
            "Road_Surface_Conditions":
                {"Dry": 1,
                 "Wet or damp": 2,
                 "Frost or ice": 3,
                 "Snow": 4,
                 "Flood over 3cm. deep": 5}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Urban_or_Rural(self, df):
        df = df[df['Urban_or_Rural_Area'] != 'Unallocated']
        encoding = {
            "Urban_or_Rural_Area":
                {"Urban": 1,
                 "Rural": 2}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Weather_Conditions(self, df):
        df = df[df['Weather_Conditions'] != 'Data missing or out of range']
        df['Weather_Conditions'] = df['Weather_Conditions'].replace(['Unknown'], 'Fine no high winds')
        encoding = {
            "Weather_Conditions":
                {"Fine no high winds": 1,
                 "Raining no high winds": 2,
                 "Raining + high winds": 3,
                 "Fine + high winds": 4,
                 "Snowing no high winds": 5,
                 "Fog or mist": 6,
                 "Snowing + high winds": 7,
                 "Other": 8}
        }
        df.replace(encoding, inplace=True)
        return df

    def handle_Speed_Limit(self, df):
        df = df[df['Speed_limit'] != 0.0]
        df.dropna(subset=['Speed_limit'], inplace=True)
        return df

    def time_to_period(self, hour):
        if 6 <= hour < 10:
            return 1
        elif 10 <= hour < 15:
            return 2
        elif 15 <= hour < 20:
            return 3
        else:
            return 4

    def handle_Time(self, df):
        df.dropna(subset=['Time'], inplace=True)
        df['Hour'] = df['Time'].str[0:2]
        df['Hour'] = pd.to_numeric(df['Hour'])
        df['Hour'] = df['Hour'].astype('int')
        df['Period'] = df['Hour'].apply(self.time_to_period)
        df.drop(columns=['Hour', 'Time'], inplace=True)
        return df

    def handle_Label(self, df):
        # convert 'Fatal' label to 'Serious'
        df['Accident_Severity'] = df['Accident_Severity'].replace(['Fatal'], 'Serious')
        df['Accident_Severity'] = df['Accident_Severity'].replace(['Serious'], 1)
        df['Accident_Severity'] = df['Accident_Severity'].replace(['Slight'], 0)
        return df

    def train_test_split(self, df):
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=100)
        return df_train, df_test

    def preprocess(self, df):
        # choose the columns we want to use for our prediction model
        # df = df[['Carriageway_Hazards',
        #          'Light_Conditions',
        #          'Day_of_Week',
        #          'Special_Conditions_at_Site',
        #          '1st_Road_Class', 'Junction_Detail',
        #          'Road_Surface_Conditions',
        #          'Urban_or_Rural_Area', 'Road_Type',
        #          'Weather_Conditions', 'Speed_limit',
        #          'Time', 'Accident_Severity']]
        # df = self.handle_Label(df)
        # df = self.downsample(df)
        df = self.handle_Carriageway_Hazards(df)
        df = self.handle_Light_Condition(df)
        df = self.handle_day_of_week(df)
        df = self.handle_Special_Conditions_at_Site(df)
        df = self.handle_1st_Road_Class(df)
        df = self.handle_Junction_Detail(df)
        df = self.handle_Road_Surface_Type(df)
        df = self.handle_Road_Type(df)
        df = self.handle_Urban_or_Rural(df)
        df = self.handle_Weather_Conditions(df)
        df = self.handle_Speed_Limit(df)
        df = self.handle_Time(df)
        df = self.handle_Label(df)
        df_train, df_test = self.train_test_split(df)
        df_train = self.downsample(df_train)
        self.update(df)
        return df_train, df_test

    def downsample(self, df):

        df_majority = df[df.Accident_Severity == 0]
        df_minority = df[df.Accident_Severity == 1]

        df_majority_downsampled = resample(df_majority,
                                           replace=False,
                                           n_samples=len(df_minority),
                                           random_state=40)

        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        return df_downsampled

    def smaller_subset(self, df_train, df_test):
        df, df_train = train_test_split(df_train, test_size=0.2, random_state=100)
        df, df_test = train_test_split(df_test, test_size=0.2, random_state=100)

        return df_train, df_test

    def get_Xy(self, df):

        y = df[self.label]
        X = df.drop(columns=self.label)
        return X, y


    def randomForestClassifier(self, df_train, df_test):

        X_train = df_train.drop(columns=self.label)
        y_train = df_train[self.label]
        X_test = df_test.drop(columns=self.label)
        y_test = df_test[self.label]

        # X_train, y_train = get_Xy(df_train)
        # X_test, y_test = get_Xy(df_test)
        # X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42)
        rf = RandomForestClassifier(bootstrap=True,
                class_weight="balanced_subsample", 
                criterion='gini',
                max_depth=8,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                min_impurity_split=None,
                min_samples_leaf=4,
                min_samples_split=10,
                min_weight_fraction_leaf=0.0,
                n_estimators=1000,
                oob_score=False,
                random_state=40,
                verbose=0,
                warm_start=False)

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        # self.modelmetrics(y_test, y_pred)
        return rf


    def Classifier(self, df_train, df_test, smallier_subset=False):
        if smallier_subset == True:
            df_train, df_test = self.smaller_subset(df_train, df_test)

        return self.randomForestClassifier(df_train, df_test)

    def predictResult(self, data):
        inputData = []
        for col in self.features:
            inputData.append(data[col])

        inputData = {'0' : inputData}
        test = pd.DataFrame.from_dict(inputData, orient='index', columns=self.features)
        new_prediction = self.model.predict(test)
        print(new_prediction)

        if new_prediction[0] == 0:
            return "Slight"
        elif new_prediction[0] == 1:
            return "Serious"
            
        return "None"



# init()

