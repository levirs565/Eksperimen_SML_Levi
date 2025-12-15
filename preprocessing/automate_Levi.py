import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
PREPROCESSED_DIR = CURRENT_DIR / "weather_preprocessing"


df = pd.read_csv(BASE_DIR / "weather_raw" / "Weather Training Data.csv").drop(
    columns=["Temp9am", "Temp3pm", "Pressure9am", "row ID"])

X, y = df.drop(columns=["RainTomorrow"]), df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

pipeline = Pipeline([
    ("impute",
        ColumnTransformer([
            ("mediam_imputer", SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number)),
            ("mode_imputer", SimpleImputer(strategy="most_frequent"),
                make_column_selector(dtype_exclude=np.number))
        ], verbose_feature_names_out=False).set_output(transform='pandas')
     ),
    ("scale_encoding",
        ColumnTransformer([
             ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [
                "Location", "WindGustDir", "WindDir9am", "WindDir3pm"]),
            ("label", OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1), ["RainToday"]),
            ("scaler", PowerTransformer(method="yeo-johnson"),
             make_column_selector(dtype_include=np.number))
        ], verbose_feature_names_out=False).set_output(transform='pandas')
     )
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(PREPROCESSED_DIR / "train.csv")
test.to_csv(PREPROCESSED_DIR / "test.csv")
