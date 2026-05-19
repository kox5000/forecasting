from flask import Flask, jsonify, request
from services.preprocessing import get_data
import traceback
import pandas as pd
from itertools import combinations
from collections import Counter

app = Flask(__name__)

@app.route("/forecast", methods=["POST"])
def get_forecast():
    data = request.get_json()
    print(data)
    try:

        result = get_data(data)
        return jsonify({
            "predictions": result,
        })
    except Exception as e:
        print("ERROR OCCURED:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400



@app.route('/insights', methods=['POST'])
def generate_insights():
    payload = request.get_json()
    data = payload
    if not data:
        return jsonify({"error": "No sales data provided"}), 400

    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])

    df['revenue'] = pd.to_numeric(df['revenue'] , errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    insights_results = []

    
    mid_point = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    filtered_df_first = df[df['date'] <= mid_point].copy()
    filtered_df_secound = df[df['date'] > mid_point].copy()


    filtered_df_first['revenue'] = filtered_df_first['revenue'].astype(float)
    filtered_df_secound['revenue'] = filtered_df_secound['revenue'].astype(float)

    # first_half = df[df['date'] <= mid_point].groupby('product_id')['revenue'].sum()
    first_half = filtered_df_first.groupby('product_id')['revenue'].sum()
    # second_half = df[df['date'] > mid_point].groupby('product_id')['revenue'].sum()
    second_half = filtered_df_secound.groupby('product_id')['revenue'].sum()
    for p_id in first_half.index:
        if p_id in second_half.index:
            f_rev = first_half[p_id]
            s_rev = second_half[p_id]
           
            if f_rev > 0:
                change = (s_rev - f_rev) / f_rev
                if change <= -0.30:
                    insights_results.append({
                        "type": "risk",
                        "title": f"Declining Sales: Product ID {int(p_id)}",
                        "description": f"Sales dropped by {abs(round(change*100))}% compared to the first half of the period."
                    })
                elif change >= 0.50:
                    insights_results.append({
                        "type": "opportunity",
                        "title": f"Growth Spike: Product ID {int(p_id)}",
                        "description": f"Significant upward trend detected (+{round(change*100)}%). Consider scaling inventory."
                    })

 
    max_date = df['date'].max()
    time_span = (df['date'].max() - df['date'].min()).days
    for p_id in df['product_id'].unique():
        product_data = df[df['product_id'] == p_id]
        last_sale_date = product_data['date'].max()
        days_inactive = (max_date - last_sale_date).days
        
  
        if days_inactive > (time_span * 0.2) and days_inactive >= 3:
            insights_results.append({
                "type": "warning",
                "title": "Dormant Product Alert",
                "description": f"Product {int(p_id)} has not seen a sale in {days_inactive} days. Check stock availability."
            })

    stats = df.groupby('product_id')['revenue'].agg(['std', 'mean']).fillna(0)
    for p_id, row in stats.iterrows():
        if row['mean'] > 50:
            cv = row['std'] / row['mean']
            if cv > 1.2:
                insights_results.append({
                    "type": "risk",
                    "title": "Unstable Sales Pattern",
                    "description": f"Product {int(p_id)} is highly volatile. This high variance makes inventory planning difficult."
                })

 
    order_groups = df.groupby('date')['product_id'].apply(list)
    pair_counter = Counter()
    for items in order_groups:
        unique_items = sorted(set(items))
        if len(unique_items) > 1:
            for pair in combinations(unique_items, 2):
                pair_counter[pair] += 1

    if pair_counter:
        most_common_pair, count = pair_counter.most_common(1)[0]
        if count >= 2: 
            insights_results.append({
                "type": "opportunity",
                "title": "Cross-Selling Opportunity",
                "description": f"Customers often buy Product {int(most_common_pair[0])} and Product {int(most_common_pair[1])} together. Consider a bundle."
            })

    
    total_batch_rev = df['revenue'].sum()
    product_totals = df.groupby('product_id')['revenue'].sum()
    for p_id, p_rev in product_totals.items():
        contribution = p_rev / total_batch_rev if total_batch_rev > 0 else 0
        if contribution < 0.05:
            insights_results.append({
                "type": "warning",
                "title": "Low Contribution",
                "description": f"Product {int(p_id)} accounts for only {round(contribution*100, 1)}% of total revenue in this dataset."
            })

    return jsonify({
        "result": {
            "insights": insights_results
        },
        "model_version": "v1.5-full-suite"
    })

@app.route('/test', methods=['POST'])
def test():
    payload = request.get_json()
    df = pd.DataFrame(payload)
    df['date'] = pd.to_datetime(df['date'])

    df['revenue'] = pd.to_numeric(df['revenue'] , errors='coerce')
    print(df.dtypes)
    return jsonify({
        "result": {
            "insights": True
        },
        "model_version": "v1.5-full-suite"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)


