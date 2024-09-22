import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

class Dataset:

    def __init__(self, name: str, model_name: str):
        self.name = name
        """ name of the dataset"""
        self.model_name = model_name
        """ model name """
        self.problem = None
        """ classification or regression """
        self.X = None
        """ X data frame """
        self.y = None
        """ y data frame """
        self.labels = None
        """ discrete label values of the dataset: classification: [0, 1]"""
        self.target_names = None
        """ name of the label: classification: ['Negative Class', 'Positive Class'], regression: ['ValueName'] """
        self.numerical_cols = None
        """ list of numerical columns which are selected after preprocessing"""
        self.categorical_cols = None
        """ list of categorical feature names which are selected after preprocessing """
        self.basic_dataset_metadata = None
        """ dictionary with basic dataset metadata """
        self.preprocessing_dataset_metadata = None
        """ dictionary with preprocessing dataset metadata """

        # Now load all these variables
        self._load_by_name()
        # self._replace_underscore()


    def _replace_underscore(self):
        self.X.columns = [col.replace('_', ' ') for col in self.X.columns]
        self.numerical_cols = [col.replace('_', ' ') for col in self.numerical_cols]
        self.categorical_cols = [col.replace('_', ' ') for col in self.categorical_cols]


    def _load_by_name(self):
        # classification data sets
        if 'water' in self.name:
            self.load_water_quality_data()
        elif 'stroke' in self.name:
            self.load_stroke_data()
        elif 'telco' in self.name:
            self.load_telco_churn_data()
        elif 'fico' in self.name:
            self.load_fico_data()
        elif 'bank' in self.name:
            self.load_bank_marketing_data()
        elif 'adult' in self.name:
            self.load_adult_data()
        elif 'airline' in self.name:
            self.load_airline_passenger_data()
        elif 'college' in self.name:
            self.load_college_data()
        elif 'weather' in self.name:
            self.load_weather_data()
        elif 'compas' in self.name:
            self.load_compas_data()
        # Regression data sets
        elif 'car' in self.name:
            self.load_car_data()
        elif 'student' in self.name:
            self.load_student_grade_data()
        elif 'crimes' in self.name:
            self.load_crimes_data()
        elif 'bike' in self.name:
            self.load_bike_sharing_data()
        elif 'housing' in self.name:
            self.load_california_housing_data()
        elif 'medical' in self.name:
            self.load_medical_data()
        elif 'crab' in self.name:
            self.load_crab_data()
        elif 'wine' in self.name:
            self.load_wine_data()
        elif 'diamond' in self.name:
            self.load_diamond_data()
        elif 'productivity' in self.name:
            self.load_productivity_data()
        elif 'mimic' in self.name:
            self.load_mimic_data()
        elif 'diabetes' in self.name: 
            self.load_diabetes_data()
        else:
            raise ValueError("Dataset not in our current dataset list! Add it to the load_datasets.py procedure!")

    def _preprocess_columns(self, df):
        """
        Drop variables with missing values >50%.
        Replace missing numerical variable values by mean.
        Replace missing categorical variable values by -1.
        Drop categorical columns with more than 25 distinct values.
        :return:
        """

        assert len(self.numerical_cols) + len(self.categorical_cols) > 0, \
            "Dataframe columns must be specified in load_datasets.py in order to preprocess them."

        # Ensure there are no empty string in numerical columns and encode as float
        df.loc[:, self.numerical_cols] = df.loc[:, self.numerical_cols].replace({'': np.nan, ' ': np.nan})

        # select columns with more than 50 % missing values
        incomplete_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
        # select categorical_cols with more than 25 unique values
        detailed_cols = df[self.categorical_cols].nunique()[df[self.categorical_cols].nunique() > 25].index.tolist()

        self.numerical_cols = [col for col in self.numerical_cols if col not in incomplete_cols]
        self.categorical_cols = [col for col in self.categorical_cols if col not in incomplete_cols and col not in detailed_cols]

        df = df.loc[:, self.numerical_cols + self.categorical_cols]

        # For categorical columns with values: fill n/a-values with -1.
        if len(self.categorical_cols) > 0:
            for categorical_col in self.categorical_cols:
                df.loc[:, categorical_col] = df.loc[:, categorical_col].fillna('unknown')
                if self.model_name == "CATBOOST":
                    df.loc[:, categorical_col] = df.loc[:, categorical_col].astype(str)
                else:
                    df.loc[:, categorical_col] = df.loc[:, categorical_col].astype('category')

        # For numerical columns with values: fill n/a-values with median.
        if len(self.numerical_cols) > 0:
            for num_col in self.numerical_cols:
                df.loc[:, num_col] = pd.to_numeric(df.loc[:, num_col], errors='coerce')
                df.loc[:, num_col] = df.loc[:, num_col].fillna(df.loc[:, num_col].median())


        # GAMens seem to have troubles with " " and "-", that why we replace them here
        def replace_char_in_string(cell_value) -> str:
            if isinstance(cell_value, str):
                for char in [" ", "-", "?", "+", "~"]:
                    cell_value = cell_value.replace(char, "_")
                return cell_value
            else:
                return cell_value

        if self.model_name == "GAMENS":
            df = df.applymap(replace_char_in_string)

        return df

    def load_water_quality_data(self):
        # 1
        # https://www.kaggle.com/adityakadiwal/water-potability
        df = pd.read_csv('data/water_potability.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
                               'Trihalomethanes', 'Turbidity']
        self.categorical_cols = []

        self.X = self._preprocess_columns(df)

        self.y = df['Potability']
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['Not Potable', 'Potable']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_college_data(self):
        # 2
        # https://www.kaggle.com/code/alrafikri/classification-potential-to-go-to-college?scriptVersionId=99597664
        df = pd.read_csv('data/college_data.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ["parent_age", "parent_salary", "house_area", "average_grades"]
        self.categorical_cols = ["type_school", "school_accreditation", "gender", "interest", "residence",
                                 "parent_was_in_college"]

        self.X = self._preprocess_columns(df)

        self.y = df['will_go_to_college']
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No College', 'College']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_stroke_data(self):
        # 3
        # https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
        df = pd.read_csv('data/healthcare-dataset-stroke-data.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        self.categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                 'Residence_type',
                                 'smoking_status']

        self.X = self._preprocess_columns(df)

        self.y = df['stroke']
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No Stroke', 'Stroke']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_telco_churn_data(self):
        # 4
        # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
        df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

        self.X = self._preprocess_columns(df)

        self.y = df['Churn']  # 'Yes, No'
        self.y = self.y.replace({'Yes': 1, 'No': 0})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No Churn', 'Churn']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_fico_data(self):
        # 5
        # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=3
        df = pd.read_csv('data/fico_heloc_dataset_v1.csv')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
                               'AverageMInFile',
                               'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                               'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades',
                               'NumTradesOpeninLast12M',
                               'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
                               'NumInqLast6Mexcl7days',
                               'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance',
                               'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
                               'PercentTradesWBalance']
        self.categorical_cols = ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
        self.X = self._preprocess_columns(df)
        # Drop rows that only contain -9. Then, all negative values will be replaced by np.nan.
        # np.nan are replaced by the column's mean in the preprocess_columns function later.
        drop_index = (self.X.eq(-9)).all(1)
        self.X = self.X[~drop_index]

        # select all columns except the target column and do conditional indexing here
        self.y = df['RiskPerformance'][~drop_index]  # 'Good', 'Bad'

        self.y = self.y.replace({'Good': 1, 'Bad': 0})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['Bad', 'Good']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_adult_data(self):
        # 6
        # https: // archive.ics.uci.edu / ml / datasets / adult
        df = pd.read_csv('data/adult_census_income.csv')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        self.categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race',
                                 'sex',
                                 'native.country']

        self.X = self._preprocess_columns(df)

        self.y = df["income"]
        self.y = self.y.replace({' <=50K': 0, ' >50K': 1})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['Income <= 50K', 'Income > 50K']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_bank_marketing_data(self):
        # 7
        # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
        df = pd.read_csv('data/bank-full.csv', sep=';')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        # TODO: Replace -1 with np.nan
        self.numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
        self.categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                 'poutcome']

        self.X = self._preprocess_columns(df)

        self.y = df['y']
        self.y = self.y.replace({'yes': 1, 'no': 0})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No subscription', 'Yes, subscribed']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_compas_data(self):
        # 8
        # https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
        df = pd.read_csv('data/compas-scores-two-years.csv')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}

        self.numerical_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                               'days_b_screening_arrest', 'c_days_from_compas']
        self.categorical_cols = ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']

        self.X = self._preprocess_columns(df)

        self.y = df['two_year_recid']

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No recidivism', 'Recidivism']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_airline_passenger_data(self):
        # 9
        # https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction
        df = pd.read_csv('data/airline_train.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                               'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                               'Seat comfort',
                               'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
                               'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
                               'Arrival Delay in Minutes']
        self.categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

        self.X = self._preprocess_columns(df)

        self.y = df['satisfaction']
        self.y = self.y.replace({'satisfied': 1, 'neutral or dissatisfied': 0})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['Neutral or dissatisfied', 'Satisfied']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_weather_data(self):
        # 10
        # https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
        df = pd.read_csv('data/weatherAUS.csv', sep=',')
        df = df.dropna(axis=0, subset='RainTomorrow')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}

        self.numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                               'WindSpeed9am',
                               'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                               'Cloud3pm',
                               'Temp9am', 'Temp3pm']
        # TODO: Später mal die Performance nochmal anschauen, weil RainToday erstmal noch mit drin. Leakage?
        self.categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

        self.X = self._preprocess_columns(df)

        self.y = df['RainTomorrow']
        self.y = self.y.replace({'No': 0, 'Yes': 1})

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['No rain tomorrow', 'Rain tomorrow']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_car_data(self):
        # 11
        # https://archive.ics.uci.edu/ml/datasets/automobile
        df = pd.read_csv('data/car.data', sep=',')
        df = df.replace("?", np.nan)
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}

        self.numerical_cols = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore',
                               'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']
        self.categorical_cols = ['symboling', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                                 'drive-wheels',
                                 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

        self.X = self._preprocess_columns(df)

        self.y = df['price']

        self.problem = 'regression'

        self.labels = 'Price'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_student_grade_data(self):
        # 12
        # https://archive.ics.uci.edu/ml/datasets/Student+Performance
        df = pd.read_csv('data/student-por.csv', sep=';')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
                               'goout',
                               'Dalc', 'Walc', 'health', 'absences']
        self.categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                                 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                                 'internet',
                                 'romantic']

        self.X = self._preprocess_columns(df)

        self.y = df['G3']

        self.problem = 'regression'
        self.labels = 'Grade'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_productivity_data(self):
        # 13
        # https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees?resource=download
        df = pd.read_csv('data/garments_worker_productivity.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
                               'no_of_style_change', 'no_of_workers']
        self.categorical_cols = ['quarter', 'department', 'day', 'team']

        self.X = self._preprocess_columns(df)

        self.y = df['actual_productivity']

        self.problem = 'regression'
        self.labels = 'Productivity'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_medical_data(self):
        # 14
        # https://www.kaggle.com/datasets/mirichoi0218/insurance
        df = pd.read_csv('data/insurance.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['age', 'bmi', 'children']
        self.categorical_cols = ['sex', 'smoker', 'region']
        self.X = self._preprocess_columns(df)
        self.y = df['charges']
        self.problem = 'regression'
        self.labels = 'Charges'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_crimes_data(self):
        # 15
        # https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
        df = pd.read_csv('data/communities.data', sep=',')
        df = df.replace('?', np.nan)
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}

        non_predictive_features = ['state', 'county', 'community', 'communityname string', 'fold', 'ViolentCrimesPerPop']
        self.numerical_cols = list(set(df.columns) - set(non_predictive_features))

        self.categorical_cols = []

        self.X = self._preprocess_columns(df)

        self.y = df['ViolentCrimesPerPop']

        self.problem = 'regression'
        self.labels = 'Violent crimes per population'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_crab_data(self):
        # 16
        # https://www.kaggle.com/datasets/sidhus/crab-age-prediction
        df = pd.read_csv('data/CrabAgePrediction.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight',
                               'Shell Weight']
        self.categorical_cols = ['Sex']

        self.X = self._preprocess_columns(df)

        self.y = df['Age']

        self.problem = 'regression'
        self.labels = 'Age'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_wine_data(self):
        # 17
        # https://archive.ics.uci.edu/ml/datasets/wine+quality
        df = pd.read_csv('data/winequality-white.csv', sep=';')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                               'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        self.categorical_cols = []

        self.X = self._preprocess_columns(df)

        self.y = df['quality']

        self.problem = 'regression'
        self.labels = 'Quality'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_bike_sharing_data(self):
        # 18
        # https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
        df = pd.read_csv('data/bike.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed', 'weekday']
        self.categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']

        self.X = self._preprocess_columns(df)

        self.y = df['cnt']

        self.problem = 'regression'
        self.labels = 'Rental Bike Count'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_california_housing_data(self):
        # 19
        # https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
        df = pd.read_csv('data/cal_housing.data', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
                               'households', 'medianIncome']
        self.categorical_cols = []

        self.X = self._preprocess_columns(df)

        self.y = df['medianHouseValue']

        self.problem = 'regression'
        self.labels = 'Median House Value'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_diamond_data(self):
        # 20
        # https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices
        df = pd.read_csv('data/Diamonds Prices2022.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
        self.categorical_cols = ['cut', 'color', 'clarity']

        self.X = self._preprocess_columns(df)

        self.y = df['price']

        self.problem = 'regression'
        self.labels = 'Price'
        self.target_names = None

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}

    def load_mimic_data(self):
        # 21
        # Src: Julian / Lasse
        df = pd.read_csv('data/IHM-complete.csv', sep=',')
        self.basic_dataset_metadata = {'n_samples': df.shape[0], 'n_columns': df.shape[1]}
        self.numerical_cols = ['Capillary refill rate+100%mean', 'Capillary refill rate+100%std', 'Diastolic blood pressure+100%mean', 'Diastolic blood pressure+100%std', 'Fraction inspired oxygen+100%mean', 'Fraction inspired oxygen+100%std', 'Glascow coma scale eye opening+100%mean', 'Glascow coma scale eye opening+100%std', 'Glascow coma scale motor response+100%mean', 'Glascow coma scale motor response+100%std', 'Glascow coma scale total+100%mean', 'Glascow coma scale total+100%std', 'Glascow coma scale verbal response+100%mean', 'Glascow coma scale verbal response+100%std', 'Glucose+100%mean', 'Glucose+100%std', 'Heart Rate+100%mean', 'Heart Rate+100%std', 'Height+100%mean', 'Height+100%std', 'Mean blood pressure+100%mean', 'Mean blood pressure+100%std', 'Oxygen saturation+100%mean', 'Oxygen saturation+100%std', 'Respiratory rate+100%mean', 'Respiratory rate+100%std', 'Systolic blood pressure+100%mean', 'Systolic blood pressure+100%std', 'Temperature+100%mean', 'Temperature+100%std', 'Weight+100%mean', 'Weight+100%std', 'pH+100%mean', 'pH+100%std']
        self.categorical_cols = []

        self.X = self._preprocess_columns(df)

        self.y = df['target']

        self.problem = 'classification'
        self.labels = [0, 1]
        self.target_names = ['Alive', 'Dead']

        self.preprocessing_dataset_metadata = {'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}
        
    def load_diabetes_data(self):
        #22 
        # Src: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
        df = pd.read_csv("data/diabetes.csv", sep=",")
        self.basic_dataset_metadata = {"n_samples": df.shape[0], "n_columns": df.shape[1]}
        self.numerical_cols = ['Age', 'Bmi' , 'Bp', 'Tc', 'Ldl', 'Hdl', 'Tch', 'Ltg', 'Glu']
        self.categorical_cols = ['Sex']
        
        self.X = self._preprocess_columns(df)
        self.y = df['Progression']
        
        self.problem = 'regression'
        self.target_names = None
        self.preprocessing_dataset_metadata={'n_samples': self.X.shape[0], 'n_columns': self.X.shape[1] + 1}
