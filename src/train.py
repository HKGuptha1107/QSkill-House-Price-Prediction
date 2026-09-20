from pathlib import Path

import pandas as pd 
import joblib
from sklearn.linear_model import LinearRegression

# Whole project path
ROOT_PATH = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
MODEL_DIR_PATH = ROOT_PATH / "models"

X_TRAIN_PATH = PROCESSED_DATA_PATH / "X_train.csv"
Y_TRAIN_PATH = PROCESSED_DATA_PATH / "y_train.csv"

MODEL_PATH = MODEL_DIR_PATH / "linear_regression.pkl"


# --- DEBUG CHECKS ---
print(f"Current Working Directory: {Path.cwd()}")
print(f"Resolved ROOT_PATH: {ROOT_PATH}")
print(f"Looking for X_train at: {X_TRAIN_PATH.resolve()}")
print(f"Does file exist? {X_TRAIN_PATH.exists()}")

if PROCESSED_DATA_PATH.exists():
    print(f"Files found in processed folder: {list(PROCESSED_DATA_PATH.iterdir())}")
else:
    print(f"Directory DOES NOT exist: {PROCESSED_DATA_PATH}")


# load data 
def load_processed_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    return X_train,y_train

# Create model 
def create_model():
    model = LinearRegression()
    return model

# train model
def train_model(model,X_train,y_train):
    model.fit(X_train,y_train.values.ravel())
    return model

# save model 
def save_model(model):
    MODEL_DIR_PATH.mkdir(exist_ok=True,parents=True)

    joblib.dump(model,MODEL_PATH)
    print("Model Saved To : ",MODEL_PATH)

# execute all methods
def main():
    print("Loading training data ...")
    X_train,y_train = load_processed_data()

    print("\nData loaded successfully ...")
    print("X_train data shape : ",X_train.shape)
    print("y_train data shape : ",y_train.shape)

    print("\nCreate linear regression model ...")

    model = create_model()

    print("\nTrain a model....")
    model = train_model(model,X_train,y_train)
    print("\nModel training completed .. ")

    print("\nTrying to save a model ..")
    save_model(model)

    print("\nTraining Model Saved Successfully ...")

if __name__ == "__main__":
    main()