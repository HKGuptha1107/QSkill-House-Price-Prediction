"""
EDA :- Exploratory Data Analysis
In this file i can do analysis in the raw data.
"""
# import neccessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# EDA Calss
class EDA:

    def __init__(self,PATH):
        self.data_path = PATH

    # load data
    def load_data(self,PATH):
        df = pd.read_csv(PATH)
        return df

    # Explore Data
    def inspect_data(self,df):
        print("Top 5 Rows : ")
        print(df.head())

        print("\nShape of DataFrame : ")
        print(df.shape)

        print("\nStatistical Description : ")
        print(df.describe())

        print("\nMissing Values : ")
        print(df.isna().sum())

        print("\nDuplicate Rows : ")
        print(df.duplicated().sum())

    def distribution_medv(self,df):
        sns.histplot("MEDV",kde=True)
        plt.title("Distribution of Median House Value !")
        plt.xlabel("MEDV")
        plt.ylabel("Frequency")
        plt.savefig("images/distibution_medv.png",dpi=300,bbox_inches="tight")
        plt.show()
        

    def features_ranges(self,df):
        df.hist(figsize=(12,10))
        plt.tight_layout()
        plt.savefig("images/features_ranges.png",dpi=300,bbox_inches="tight")
        plt.show()

    def feature_compare_target(self,df,target="MEDV"):
        col_names = [col for col in df.columns if col != target]
        num_plots = len(col_names)

        num_cols = 3
        num_rows = math.ceil(num_plots/num_cols)

        fig,axes = plt.subplots(
            nrows=num_rows,ncols=num_cols,figsize=(10, 4 * num_rows)
        )

        axes = axes.flatten()

        for idx,col in enumerate(col_names):
            sns.scatterplot(data=df,x=col,y=target,ax=axes[idx])
            axes[idx].set_title(f"{col} vs {target}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel(target)
        plt.tight_layout()
        plt.savefig("images/feature_compare_target.png",dpi=300,bbox_inches="tight")
        plt.show()

    def inspect_outliers(self,df):
        col_name = ["CRIM","ZN","DIS","RM","LSTAT","MEDV"]
        num_plots = len(col_name)

        num_cols = 3
        num_rows = math.ceil(num_plots/num_cols)

        fig,axes = plt.subplots(
            nrows=num_rows,ncols=num_cols,figsize=(15, 4 * num_rows)
        )
        axes = axes.flatten()
        for idx,col in enumerate(col_name):
            sns.boxplot(data=df,x=col,ax=axes[idx])
            axes[idx].set_title(col)
        plt.tight_layout()
        plt.savefig("images/inspect_outliers.png",dpi=300,bbox_inches="tight")
        plt.show()

    def undstd_rel_num(self,df):
        correlation = df.corr(numeric_only=True)
        sns.heatmap(correlation,annot=True,cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("images/understand_rel_numeric.png",dpi=300,bbox_inches="tight")
        plt.show()

    def main(self):
        df  = self.load_data(self.data_path)
        self.inspect_data(df)
        self.distribution_medv(df)
        self.features_ranges(df)
        self.feature_compare_target(df,)
        self.inspect_outliers(df)
        self.undstd_rel_num(df)
if __name__ == '__main__':
    """ls = ["data/processed/X_test.csv","data/processed/X_train.csv","data/processed/y_test.csv","data/processed/y_train.csv"]
    for i in ls:
        obj = EDA(i)
        obj.main()"""

    PATH = "data/raw/HousingData.csv"
    obj = EDA(PATH)
    obj.main()