
"""
#  Preprocess the data for training and testing 
# This script is responsible for loading the Boston Housing dataset, performing necessary preprocessing steps such as handling missing values, encoding categorical variables, and splitting the data into training and testing sets. The processed data will be saved to specified file paths for later use in model training and evaluation. 
# necessary libraries are imported, and the dataset is loaded from the specified path. The preprocessing steps are applied, and the resulting training and testing datasets are saved to the designated file paths.
"""

import pandas as pd 
from sklearn.model_selection import train_test_split 

# create a preprocesing class to handle the preprocessing of the data

class Preprocessor:
    # Load dataset 
    def __init__(self,dataset_path):
        self.dataset = dataset_path

    # convert dataset into dataFrame which is used in pandas
    def load_data(self):
        df = pd.read_csv(self.dataset)
        return df 
    
    # understand DataSet 
    def inspect_data(self,data):
        df = data
        print("Top 5 rows of a HousingData.csv :\n",df.head())
        print("\n\nDataSet Shape : ",df.shape)
        print("\nDataSet Information : \n")
        df.info()
        print("\nDataSet Description (like count,mean,standard deviation,minimum,maximum,etc..) : ")
        print("\n",df.describe())

    # Check the missing values in the dataset
    def handling_missing_values(self,data):
        df = data
        
        # Check which data missing in enitre dataframe
        if df.isna().any().any():
            print("The DataSet having missing values !")

            # Missing Count will filter the each column seperatly which column having greater than 0 missing values
            missingCount = df.isna().sum()
            print(f"DataSet Missing values Count :\n{missingCount[missingCount > 0]}")
            print("\nNow missing values are filled with there respective columns mean values ....")
            # select only numeric columns for applying the mean 
            numericCols = df.select_dtypes(include=['number']).columns

            # fill values with there respective columns means
            df[numericCols] = df[numericCols].fillna(df[numericCols].mean())

            print("Successfully fill missing values !")
        else:
            print("These DataFrame is not having a Missing Values !")
        return df

    # Remove the duplicate of data in the dataframe 
    def remove_duplicate(self,data):
        df = data

        # count duplicate rows in dataframe
        duplicateCount = df.duplicated().sum()

        if duplicateCount > 0:
            print(f"Duplicate rows found count : {duplicateCount}")
            print("These Dataframe having duplicate rows so now ")
            df = df.drop_duplicates()
        return df

    # seperate the features target
    def seperate_feature_target(self,df):
        X = df.drop("MEDV",axis=1)
        y = df["MEDV"]

        print("Features : ")
        print(X.head())

        print("Target : ")
        print(y.head())
        return X,y

    # split data into training and testing
    def split_data(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
        print("\nTraining Data Shape : ")
        print(X_train.shape)

        print("\nTesting Data Shape : ")
        print(y_test.shape)
        return X_train,X_test,y_train,y_test

    # save the preprocessed data with csv format
    def save_preprocessed_data(self,X_train,X_test,y_train,y_test):
        X_train.to_csv("data/processed/X_train.csv",index=False)
        X_test.to_csv("data/processed/X_test.csv",index=False)
        y_train.to_csv("data/processed/y_train.csv",index=False)
        y_test.to_csv("data/processed/y_test.csv",index=False)
        print("\nSucessfully Saved all Preprocessed data in (data/processed/..) ")

    def main(self):
        # Exceute all methods one by one here
        # 1. Load DataSet 
        df = self.load_data()
        # 2. UnderStand Data what datatypes, info of data etc..
        self.inspect_data(df)
        # 3. remove duplicates
        df = self.remove_duplicate(df)
        # 4. first we find the missing values then fill missing data
        df = self.handling_missing_values(df)
        # 5. seperate featuers from  data 
        X,y = self.seperate_feature_target(df)
        # 6. split data into X_train,x_test,y_train,y_test
        X_train,X_test,y_train,y_test = self.split_data(X,y)
        # 7. Save preocesssed data
        self.save_preprocessed_data(X_train,X_test,y_train,y_test)
        print("\n\n\t\t Successfully PreProcessing Completed !")
        
if __name__ == "__main__":
    DATA_SET_PATH = "data/raw/HousingData.csv"
    obj = Preprocessor(DATA_SET_PATH)
    obj.main()