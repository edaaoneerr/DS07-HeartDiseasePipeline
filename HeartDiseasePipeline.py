import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Load data
df = pd.read_csv("heart_disease_uci.csv")
# Print all column names to verify if 'num' exists

################################################
# age: Age of the patient.
# sex: Gender of the patient.
# cp (chest pain type): Type of chest pain experienced.
# trestbps: Resting blood pressure (in mm Hg).
# chol: Serum cholesterol in mg/dl.
# fbs (fasting blood sugar): Whether fasting blood sugar is greater than 120 mg/dl (Boolean).
# restecg: Resting electrocardiographic results.
# thalch: Maximum heart rate achieved.
# exang: Exercise induced angina (Boolean).
# oldpeak: ST depression induced by exercise relative to rest.
# slope: Slope of the peak exercise ST segment.
# ca: Number of major vessels colored by fluoroscopy.
# thal: Thalassemia type.
# num: Diagnosis of heart disease (angiographic disease status).
################################################

# Initial data check
print("Initial missing values:", df.isnull().sum().sum())
print(df.columns)

df.head()
df.isnull().sum()
df.info()
df.describe().T


# Split data
X = df.drop("num", axis=1)  # Assuming 'num' is the target
y = df[["num"]]

def preprocess_pipeline(numeric_features, categorical_features):
    """Create a preprocessing pipeline."""

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

# Define features
numeric_features = df.select_dtypes(include=['int64', 'float64'], exclude=[]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Setup preprocessing steps
preprocessor = preprocess_pipeline(numeric_features, categorical_features)

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.columns, X_test.columns, y_train.columns, y_test.columns)
# Train and evaluate the model
pipeline.fit(X_train, y_train)
print("Model training complete.")
print("Model score:", pipeline.score(X_test, y_test))
print(classification_report(y_test, pipeline.predict(X_test)))

# Check for any remaining missing values
print("Missing values after preprocessing:", X_train.isnull().sum().sum())


# Define features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Setup preprocessing steps
preprocessor = preprocess_pipeline(numeric_features, categorical_features)

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])

def check_missing_values(dataframe):
    print("Eksik Değer Analizi:")
    print(dataframe.isnull().sum())


def fill_missing_values(dataframe):
    # Numeric columns: impute with mean
    num_imputer = SimpleImputer(strategy='mean')
    num_cols = dataframe.select_dtypes(include=['number']).columns  # 'number' covers both int and float
    dataframe[num_cols] = num_imputer.fit_transform(dataframe[num_cols])

    # Categorical columns: impute with the most frequent value or a placeholder
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[cat_cols] = cat_imputer.fit_transform(dataframe[cat_cols])

    return dataframe

def check_and_clean_data(dataframe):
    print("Checking for NaN values after imputation:")
    if dataframe.isnull().sum().any():
        print("NaN values detected. Filling missing values again.")
        dataframe = fill_missing_values(dataframe)
    else:
        print("No NaN values detected.")
    return dataframe


def heart_disease_data_prep(dataframe):
    check_missing_values(dataframe)
    dataframe = fill_missing_values(dataframe)
    dataframe = check_and_clean_data(dataframe)  # Verify and clean again if needed

    # One-hot encode categorical variables
    categorical_cols = ['sex', 'cp', 'thal', 'slope', 'dataset', 'restecg']
    dataframe = one_hot_encoder(dataframe, categorical_cols)

    # Outlier handling
    num_cols = list(dataframe.select_dtypes(include=['number']).columns)
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    X = dataframe.drop("num", axis=1)
    y = dataframe["num"]
    X = StandardScaler().fit_transform(X)
    return X, y


################################################
# Helper Functions
################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Ensure the column is float to handle continuous outlier limits
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def encode_features(dataframe):
    label_encoders = {}
    for column in ['sex']:
        le = LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column])
        label_encoders[column] = le
    return dataframe, label_encoders

def heart_disease_data_prep(dataframe):
    # Tüm kategorik sütunları one-hot encode ile işle
    categorical_cols = ['sex', 'cp', 'thal', 'slope', 'dataset', 'restecg']  # Tüm kategorik sütunları listele
    dataframe = one_hot_encoder(dataframe, categorical_cols)

    num_cols = list(dataframe.select_dtypes(include=['int64', 'float64']).columns)
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
    X = dataframe.drop("num", axis=1)
    y = dataframe["num"]
    X = StandardScaler().fit_transform(X)
    return X, y

# Base Models
def base_models(X, y, scoring="roc_auc"):
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier())]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# Hyperparameter Optimization
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    classifiers = [('KNN', KNeighborsClassifier(), {"n_neighbors": range(2, 50)}),
                   ("CART", DecisionTreeClassifier(), {'max_depth': range(1, 20), "min_samples_split": range(2, 30)}),
                   ("RF", RandomForestClassifier(), {"max_depth": [8, 15, None], "max_features": [5, 7, "auto"],
                                                     "min_samples_split": [15, 20], "n_estimators": [200, 300]}),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {"learning_rate": [0.1, 0.01],
                                                                                            "max_depth": [5, 8],
                                                                                            "n_estimators": [100, 200],
                                                                                            "colsample_bytree": [0.5, 1]}),
                   ('LightGBM', LGBMClassifier(), {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500],
                                                   "colsample_bytree": [0.7, 1]})]
    best_models = {}
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)} ({name}) Best Params: {gs_best.best_params_}")
        best_models[name] = final_model
    return best_models

# Voting Classifier
def voting_classifier(best_models, X, y):
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf






################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("heart_disease_uci.csv")
    X, y = heart_disease_data_prep(df)
    print("Base model results:")
    base_models(X, y)
    print("Optimizing models...")
    best_models = hyperparameter_optimization(X, y)
    print("Building voting classifier...")
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    print("Process completed successfully!")


if __name__ == "__main__":
    main()
