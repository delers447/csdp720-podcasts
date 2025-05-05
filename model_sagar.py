#%%
from itertools import combinations
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm
#%%
def preprocess_data(df):
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    df.drop(columns=['Episode_Title'])

    categorical_cols = ['Episode_Num', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for cat_col in categorical_cols:
        df[cat_col] = df[cat_col].astype('category')

    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype('int')
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=False)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)

    return df
#%%
def evaluate_model(X, y, df_test, model_params, n_splits=5):
    kfold = KFold(n_splits= n_splits, shuffle=True, random_state=42)

    y_pred = np.zeros(len(df_test))
    oof = np.zeros(len(X))
    rmse_scores = []

    for fold, (idx_train, idx_valid) in enumerate(kfold.split(X, y)):

        print("------------------------------------------------------------------------------")
        print(f"Fold {fold+1} Processing")
        print("------------------------------------------------------------------------------")

        X_train = X.iloc[idx_train].copy()
        X_valid = X.iloc[idx_valid].copy()

        y_train = y.iloc[idx_train].copy()
        y_valid = y.iloc[idx_valid].copy()

        x_test = df_test[X.columns].copy()

        encoded_columns = X.columns[11:]
        for c in tqdm(encoded_columns, desc="Encoding Columns"):
            encoder = TargetEncoder(smooth=0)

            X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
            X_valid[c] = encoder.transform(X_valid[[c]])
            x_test[c] = encoder.transform(x_test[[c]])

            model = LGBMRegressor(**model_params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    early_stopping(stopping_rounds=2500),
                    log_evaluation(period=500)
                ]
            )

            oof[idx_valid] = model.predict(X_valid)
            y_pred += model.predict(x_test)

            fold_rmse = mean_squared_error(y_valid, oof[idx_valid])
            rmse_scores.append(fold_rmse)

            print("--------------------------------------------------------------------------------------")
            print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
            print("---------------------------------------------------------------------------------------")

    rmse = mean_squared_error(y, oof)
    print("==================================================================================================")
    print(f"Overall CV RMSE {rmse:.5f}")
    print("==================================================================================================")

    y_pred /= n_splits
    return y_pred, oof, rmse, model

#%%
# load the data

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)
#%%
X = train_df.drop(columns=['id', 'Episode_Title', 'Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']
#%%
model_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'rmse',
    'random_state': 42,
    'device': 'gpu',
    'gpu_platform_id': 2,
    'gpu_device_id': 2
}

y_pred_result, oof_result, rmse_result, model = evaluate_model(X, y, test_df, model_params)
#%%
df_sam = pd.read_csv('./data/sample_submission.csv')
#%%
df_sam['Listening_Time_minutes'] = y_pred_result
#%%
df_sam.head()
#%%
model.booster_.save_model("sagar_train.txt")
#%%
import joblib as jl
jl.dump(model, "model_sagar.pkl")
#%%
