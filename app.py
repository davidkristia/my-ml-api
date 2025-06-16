from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load model dan scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load data awal dari CSV
CSV_PATH = 'tourism_with_predicted_clusters.csv'
df = pd.read_csv(CSV_PATH)

# Simpan rencana yang ditambahkan secara manual
added_plans = []

@app.route('/')
def home():
    return jsonify({
        'message': '✅ ML Wisata API is running!',
        'endpoints': {
            'POST /predict-cluster': 'Prediksi cluster dari input user',
            'GET /get-recommendations/<cluster_id>': 'Ambil rekomendasi dari cluster',
            'GET /generate-itinerary/<cluster_id>': 'Buat itinerary otomatis dari cluster',
            'POST /add-plan/<cluster_id>': 'Tambah rencana wisata ke cluster tertentu',
            'GET /plans/<cluster_id>': 'Ambil semua rencana perjalanan dari cluster',
            'GET /get-added-plans': 'Ambil semua rencana user yang ditambahkan'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'OK'})

@app.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        data = request.get_json()
        rating = float(data.get('Rating', 0))
        price = float(data.get('Price', 0))
        time = float(data.get('Time_Minutes', 0))
        lat = float(data.get('Lat', 0))
        long = float(data.get('Long', 0))

        features = [[rating, price, time, lat, long]]
        scaled = scaler.transform(features)
        cluster = int(kmeans.predict(scaled)[0])

        return jsonify({'cluster': cluster})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get-recommendations/<int:cluster_id>', methods=['GET'])
def get_recommendations(cluster_id):
    try:
        recommendations = df[df['Predicted_Cluster'] == cluster_id][[
            'Place_Name', 'Category', 'City', 'Rating', 'Price'
        ]].to_dict(orient='records')

        return jsonify({
            'cluster': cluster_id,
            'total_recommendations': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-itinerary/<int:cluster_id>', methods=['GET'])
def generate_itinerary(cluster_id):
    try:
        places = df[df['Predicted_Cluster'] == cluster_id][[
            'Place_Name', 'Category', 'City', 'Rating', 'Price'
        ]]

        if places.empty:
            return jsonify({'error': 'Cluster tidak ditemukan'}), 404

        itinerary = places.sort_values(by='Rating', ascending=False).head(5).to_dict(orient='records')

        return jsonify({
            'cluster': cluster_id,
            'itinerary_count': len(itinerary),
            'itinerary': itinerary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-plan/<int:cluster_id>', methods=['POST', 'OPTIONS'])
def add_plan(cluster_id):
    try:
        data = request.get_json()

        place_name = data.get('Place_Name')
        category = data.get('Category')
        city = data.get('City')
        rating = float(data.get('Rating', 0))
        price = int(data.get('Price', 0))

        if not all([place_name, category, city]):
            return jsonify({'error': 'Field Place_Name, Category, dan City wajib diisi'}), 400

        new_plan = {
            'cluster': cluster_id,
            'Place_Name': place_name,
            'Category': category,
            'City': city,
            'Rating': rating,
            'Price': price
        }

        added_plans.append(new_plan)

        return jsonify({
            'message': '✅ Rencana wisata berhasil ditambahkan',
            'data': new_plan
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plans/<int:cluster_id>', methods=['GET'])
def get_plans_by_cluster(cluster_id):
    try:
        filtered = [plan for plan in added_plans if int(plan['cluster']) == cluster_id]
        return jsonify({
            'cluster': cluster_id,
            'total_plans': len(filtered),
            'plans': filtered
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-added-plans', methods=['GET'])
def get_all_added_plans():
    try:
        return jsonify({'plans': added_plans})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
