import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import boto3
from datetime import datetime
from decimal import Decimal
import numpy as np

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket = "amazon-sagemaker-736116164611-us-east-2-ayappcqas719d5"
    key = "data/train19e.csv"
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])

    # Target and features
    y = df["Target_Open_NVDA"].iloc[:-1]
    X = df.drop(columns=["Target_Open_NVDA"])
    X = X.iloc[:-1]

    # Convert date to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    X['day'] = df['Date'].dt.day
    X['month'] = df['Date'].dt.month
    X['year'] = df['Date'].dt.year

    # Drop original Date column
    X = X.drop(columns=["Date"])

    # XGBoost Model
    # -------------------------------
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        objective="reg:squarederror",
        reg_lambda=1,
        colsample_bytree=0.9,
        reg_alpha=0.3
    )
    model.fit(X, y)

    s3_key = "model/nvda_model.json"
    model.save_model("/tmp/nvda_model.json")
    s3.upload_file("/tmp/nvda_model.json", bucket, s3_key)

    # Prepare the latest row for prediction
    X_new = df.iloc[[-1]].copy()  # keep as DataFrame

    # Extract numeric date features
    X_new['day'] = X_new['Date'].dt.day
    X_new['month'] = X_new['Date'].dt.month
    X_new['year'] = X_new['Date'].dt.year
    X_new['dayofweek'] = X_new['Date'].dt.dayofweek

    # Drop original Date column
    X_new = X_new.drop(columns=['Date', 'Target_Open_NVDA'])
    X_new = X_new[X.columns]
    pred_open = model.predict(X_new)[0]

    # -------------------------------
    # Monte Carlo simulation for low/high range
    # -------------------------------
    N = 1000          # number of simulations
    factor = 0.5      # fraction of std to use

    # Historical volatility per feature
    feature_stds = X.std()
    importance = model.get_booster().get_score(importance_type='gain')
    max_imp = max(importance.values())
    min_noise = 0.01
    scaled_importance = {k: max(importance.get(k, 0)/max_imp, min_noise) for k in X.columns}

    X_base = X_new.iloc[0].copy()
    X_sim = pd.DataFrame([X_base]*N, columns=X_new.columns)

    for col in X_sim.columns:
        hist_std_noise = factor * feature_stds[col]
        importance_scale = scaled_importance[col]
        total_noise = hist_std_noise * importance_scale
        X_sim[col] += np.random.normal(0, total_noise, size=N)

    preds_mc = model.predict(X_sim)
    ci_lower, ci_upper = np.percentile(preds_mc, [5, 95])  # 95% CI as low/high range

    # -------------------------------
    # Save prediction and range to DynamoDB
    # -------------------------------
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table("NVDA_Predictions")
    prediction_date = datetime.now().strftime("%Y-%m-%d")
    item = {
        "PredictionDate": prediction_date,  # partition key
        "Predicted_Open_NVDA": Decimal(str(pred_open)),
        "Predicted_Low_NVDA": Decimal(str(ci_lower)),
        "Predicted_High_NVDA": Decimal(str(ci_upper))
    }
    table.put_item(Item=item)

    return {
        "status": "success",
        "Predicted_Open_NVDA": float(pred_open),
        "Predicted_Low_NVDA": float(ci_lower),
        "Predicted_High_NVDA": float(ci_upper)
    }
