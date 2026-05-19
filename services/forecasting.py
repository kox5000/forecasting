import pandas as pd
import xgboost as xgb
import os
from datetime import timedelta

model = xgb.XGBRegressor()

model.load_model("trained_model/model.json")

def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn


def run_model(data_list):
    all_predictions = []

   
    if not isinstance(data_list, list):
        data_list = [data_list]

    for item in data_list:
    
        dom = item['day_of_month']
        m = item['month']
        dn = item['day_num']
        sn = item['season_num']
        history = item['last_sales_list']
        
      
        sales_lag_1 = history[-1]
        sales_lag_7 = history[0]
        rolling_mean_7 = sum(history) / 7
        
    
        input_df = pd.DataFrame([[
            dom, m, dn, sn, sales_lag_1, sales_lag_7, rolling_mean_7
        ]], columns=['day_of_month', 'month', 'day_num', 'season_num', 
                     'sales_lag_1', 'sales_lag_7', 'rolling_mean_7'])
        
    
        pred = model.predict(input_df)[0]
        
     
        all_predictions.append({
            "date_info": f"Day {dom}/Month {m}",
            "prediction": round(float(pred), 2)
        })

    return all_predictions


def run_model_for_future(df):
    predictions = []
    
 
    last_date = df['date'].max()
    

    current_history = df['quantity'].tolist()

    for i in range(1, 8): 
        prediction_date = last_date + timedelta(days=i)
        
    
        dom = prediction_date.day
        m = prediction_date.month
        dn = (prediction_date.weekday() + 1) % 7
        sn = get_season(m)
        
    
        sales_lag_1 = current_history[-1] if len(current_history) >= 1 else 0
        sales_lag_7 = current_history[-7] if len(current_history) >= 7 else 0
        rolling_mean_7 = sum(current_history[-7:]) / 7 if len(current_history) >= 7 else sum(current_history)/len(current_history)

       
        input_df = pd.DataFrame([[
            dom, m, dn, sn, sales_lag_1, sales_lag_7, rolling_mean_7
        ]], columns=['day_of_month', 'month', 'day_num', 'season_num', 
                     'sales_lag_1', 'sales_lag_7', 'rolling_mean_7'])


        pred = model.predict(input_df)[0]
        pred = max(0, float(pred)) 
        
    
        current_history.append(pred)
        
        predictions.append({
            "date": prediction_date.strftime("%Y-%m-%d"),
            "prediction": round(pred, 2)
        })

    return predictions