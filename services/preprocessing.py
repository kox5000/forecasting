from services.forecasting import run_model
from services.forecasting import run_model_for_future
from datetime import datetime
from datetime import timedelta
import pandas as pd


def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn


def get_data(data):
    sales_data = data.get("sales_data", [])
    if not sales_data:
        return []


    df = pd.DataFrame(sales_data)
    df['date'] = pd.to_datetime(df['date'])
    df['quantity'] = df['quantity'].astype(float)
    
   
    df = df.groupby('date')['quantity'].sum().reset_index()


    df = df.set_index('date').asfreq('D', fill_value=0).reset_index()
    

    df = df.sort_values('date')


    result = run_model_for_future(df)

    return result