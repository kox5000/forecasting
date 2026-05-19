# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os

# df = pd.read_csv("datasets/sales_dataset_learning_ready.csv")

# df['sales_lag_1'] = df['sales'].shift(1)
# df['sales_lag_7'] = df['sales'].shift(7)
# df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()

# df.dropna(inplace=True)

# features = ['day_of_month', 'month', 'day_num', 'season_num', 
#             'sales_lag_1', 'sales_lag_7', 'rolling_mean_7']

# x = df[features]
# y = df['sales']

# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, shuffle=False)


# model = xgb.XGBRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=5,
#     early_stopping_rounds=50
# )

# model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# y_pred = model.predict(X_test)

# # اسم الفولدر اللي عايز تحفظ فيه
# folder_path = "trained_model"

# # لو الفولدر مش موجود.. الكود هيعمله فوراً
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# # دلوقتي احفظ وأنت مطمن
# model.save_model(os.path.join(folder_path, "model.json"))

# print(f"✅ Model saved successfully in {folder_path}/model.json")

# plt.figure(figsize=(12, 6))
# plt.plot(y_test.values, label='Actual Sales', color='blue')
# plt.plot(y_pred, label='Predicted Sales', color='red', linestyle='--')
# plt.title('Actual vs Predicted Sales')
# plt.legend()
# plt.show()

# xgb.plot_importance(model)
# plt.show()

# print(f"R2 Score: {model.score(X_test, y_test):.4f}")