from datetime import date
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""python preprocess.py \n
    --pathToCSV data/marketing_campaign.csv \n
    --pathToSaveProcessedCSV data/processed_marketing_campaign.csv"""


def get_Age(birth_year):
    return date.today().year - int(birth_year)



 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathToCSV', type=str, default='marketing_campaign.csv')
    parser.add_argument("--pathToSaveProcessedCSV", type=str, default='processed_marketing_campaign.csv')

    args = parser.parse_args()

    
    df = pd.read_csv(args.pathToCSV, sep='\t')
    print("---Started Preprocessing---")
    print("\n\nDropping rows having null values")
    df.dropna(inplace=True)
    print("Extracting new feature from Birth Year feature")
    df['Age'] = df['Year_Birth'].apply(get_Age)
    print(df.head())
    print('\n\n Creating new feature total spent from spent features ')
    df['TotalSpent'] =  df.MntWines + df.MntFruits + df.MntMeatProducts + df.MntFishProducts + df.MntSweetProducts + df.MntGoldProds
    print(df.head())
    print("\n \nExtracing Months spent on the platform from feature Dt_Customer")
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Mn_Customer'] = 12.0 * (2015 - df.Dt_Customer.dt.year ) + (1 - df.Dt_Customer.dt.month)
    print(df.head())
    print("\n \nGrouping customer based on Age \n")
    df.loc[(df['Age'] >= 13) & (df['Age'] <= 19), 'AgeGroup'] = 'Teen'
    df.loc[(df['Age'] >= 20) & (df['Age']<= 39), 'AgeGroup'] = 'Adult'
    df.loc[(df['Age'] >= 40) & (df['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
    df.loc[(df['Age'] > 60), 'AgeGroup'] = 'Senior Adult'
    print(df.head())
    print("\n \n Creating new feature from kidHome and teenHome \n")
    df['Children'] = df['Kidhome'] + df['Teenhome']
    print(df.head())
    print("\n \nUpdating martial_status to single or married")
    df.Marital_Status = df.Marital_Status.replace({'Together': 'Married',
                                                    'Married': 'Married',
                                                    'Divorced': 'Single',
                                                    'Widow': 'Single', 
                                                    'Alone': 'Single',
                                                    'Absurd': 'Single',
                                                    'YOLO': 'Single'})
    print(df.head())
    print("\n\nChecking and dropping rows having absurd age values like greater than 100")
    df = df[df.Age < 95]
    df = df[(df["Income"]<600000)]
    print("\nRows with ages more than 100 dropped")
    print(df.Education.value_counts())
    print("\n Replacing 2n cycle in Education column to Master")
    df['Education'] = df['Education'].str.replace('2n Cycle', 'Master') 
    print(df.Education.value_counts())
    df['campaign'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
    df['campaignConvert'] = df['campaign'].apply(lambda x: 1 if x else 0)
    print('\n\nDropping some redundant columns')
    df.drop(["Marital_Status", 'campaign', "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID", 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response'], axis=1, inplace=True)
    print(df.head())
    # Label Encoding and Standardizing features
    print("\n\nLabel Encoding categorical features")
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    le = LabelEncoder()
    print(f'Categorical columns are {object_cols}')
    for col in object_cols:
        df[col] = df[[col]].apply(le.fit_transform)
    print("\n\nAll categorical features have been label encoded")
    print(df.dtypes)
    print("\n\nStandardizing features ...")
    col = df.campaignConvert
    scaler = StandardScaler()
    df = df[['Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
       'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age',
       'TotalSpent', 'Mn_Customer', 'AgeGroup', 'Children']]
    scaler.fit(df)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=['Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
       'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age',
       'TotalSpent', 'Mn_Customer', 'AgeGroup', 'Children'])
    print("All features have been scaled")
    scaled_df['campaignConvert'] = col
    scaled_df['campaignConvert'] =scaled_df['campaignConvert'].apply(lambda x: 1 if x == 1.0 else 0)
    print(scaled_df.head())

    print("\n\nPreprocessing done")
    scaled_df.to_csv(args.pathToSaveProcessedCSV,index=None)
    print(f"Processed file saved at {args.pathToSaveProcessedCSV}")


