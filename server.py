"""
=============================================================================
TELCO CUSTOMER CHURN - LIVE ANALYTICS & ML INFERENCE HTTP SERVER
Author: Simran Karmali (Enhanced & Productionized)
Description: Python HTTP server serving the interactive web dashboard, real-time
             Machine Learning inference API, and in-memory SQL execution studio.
=============================================================================
"""

import os
import sys
import json
import time
import sqlite3
import joblib
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')
IMG_DIR = os.path.join(BASE_DIR, 'images')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Customer-Churn_Telco_Clean.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'python', 'models', 'best_churn_model.pkl')

PORT = 8000

# ---------------------------------------------------------------------------
# Global In-Memory Model & SQLite Initialization
# ---------------------------------------------------------------------------
print(f"--> Loading ML Pipeline from: {MODEL_PATH}")
ML_MODEL = joblib.load(MODEL_PATH)
print("--> ML Pipeline loaded successfully.")

print("--> Initializing In-Memory SQLite Database...")
SQL_CONN = sqlite3.connect(':memory:', check_same_thread=False)
DF_CLEAN = pd.read_csv(DATA_PATH)
if 'Unnamed: 0' in DF_CLEAN.columns:
    DF_CLEAN = DF_CLEAN.drop(columns=['Unnamed: 0'])
DF_CLEAN.to_sql('telco_clean', SQL_CONN, index=False, if_exists='replace')
print(f"--> Loaded {len(DF_CLEAN)} records into 'telco_clean' table.")


class ChurnServerHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Override to keep logs clean
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]}\n")

    def send_json_response(self, status_code, data):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Root route: serve web/index.html
        if path == '/' or path == '/index.html':
            index_file = os.path.join(WEB_DIR, 'index.html')
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "index.html not found")
                return

        # Serve static images from /images/
        elif path.startswith('/images/'):
            filename = os.path.basename(path)
            img_path = os.path.join(IMG_DIR, filename)
            if os.path.exists(img_path) and os.path.isfile(img_path):
                ext = os.path.splitext(filename)[1].lower()
                content_type = 'image/png' if ext == '.png' else 'image/jpeg'
                with open(img_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Image not found")
                return

        # API: Metrics summary
        elif path == '/api/metrics':
            total = len(DF_CLEAN)
            churned = int((DF_CLEAN['Churn'] == 'Yes').sum())
            retained = total - churned
            churn_rate = round(100.0 * churned / total, 2)
            total_mrr = round(float(DF_CLEAN['MonthlyCharges'].sum()), 2)
            lost_mrr = round(float(DF_CLEAN[DF_CLEAN['Churn'] == 'Yes']['MonthlyCharges'].sum()), 2)
            annual_lost = round(lost_mrr * 12, 2)

            data = {
                'total_customers': total,
                'churned_customers': churned,
                'retained_customers': retained,
                'churn_rate_pct': churn_rate,
                'total_mrr': total_mrr,
                'lost_mrr': lost_mrr,
                'annualized_lost_mrr': annual_lost
            }
            self.send_json_response(200, data)
            return

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_length).decode('utf-8')

        try:
            req_data = json.loads(post_body) if post_body else {}
        except json.JSONDecodeError:
            self.send_json_response(400, {'error': 'Invalid JSON body'})
            return

        # Endpoint: Real-Time ML Churn Prediction
        if path == '/api/predict':
            try:
                # Required feature order for the pipeline preprocessor
                sample_df = pd.DataFrame([{
                    'gender': req_data.get('gender', 'Male'),
                    'SeniorCitizen': int(req_data.get('SeniorCitizen', 0)),
                    'Partner': req_data.get('Partner', 'No'),
                    'Dependents': req_data.get('Dependents', 'No'),
                    'tenure': int(req_data.get('tenure', 1)),
                    'PhoneService': req_data.get('PhoneService', 'Yes'),
                    'MultipleLines': req_data.get('MultipleLines', 'No'),
                    'InternetService': req_data.get('InternetService', 'Fiber optic'),
                    'OnlineSecurity': req_data.get('OnlineSecurity', 'No'),
                    'OnlineBackup': req_data.get('OnlineBackup', 'No'),
                    'DeviceProtection': req_data.get('DeviceProtection', 'No'),
                    'TechSupport': req_data.get('TechSupport', 'No'),
                    'StreamingTV': req_data.get('StreamingTV', 'No'),
                    'StreamingMovies': req_data.get('StreamingMovies', 'No'),
                    'Contract': req_data.get('Contract', 'Month-to-month'),
                    'PaperlessBilling': req_data.get('PaperlessBilling', 'Yes'),
                    'PaymentMethod': req_data.get('PaymentMethod', 'Electronic check'),
                    'MonthlyCharges': float(req_data.get('MonthlyCharges', 70.0)),
                    'TotalCharges': float(req_data.get('TotalCharges', 140.0))
                }])

                prob = float(ML_MODEL.predict_proba(sample_df)[0][1])
                pred = int(ML_MODEL.predict(sample_df)[0])

                # Determine risk level and drivers
                if prob >= 0.65:
                    risk_level = 'High Risk'
                elif prob >= 0.40:
                    risk_level = 'Moderate Risk'
                else:
                    risk_level = 'Low Risk'

                drivers = []
                if sample_df['Contract'].iloc[0] == 'Month-to-month':
                    drivers.append('Month-to-Month Contract (+35% Risk)')
                if sample_df['InternetService'].iloc[0] == 'Fiber optic':
                    drivers.append('Fiber Optic Price-to-Value Friction (+22% Risk)')
                if sample_df['PaymentMethod'].iloc[0] == 'Electronic check':
                    drivers.append('Electronic Check Manual Billing (+20% Risk)')
                if sample_df['tenure'].iloc[0] <= 12:
                    drivers.append('First-Year Onboarding Hazard (+25% Risk)')
                if sample_df['TechSupport'].iloc[0] == 'No':
                    drivers.append('Lack of Tech Support Add-on (+18% Risk)')
                if not drivers:
                    drivers.append('Long Tenure & Stable Long-Term Contract (-60% Risk)')

                # Recommended action
                if risk_level == 'High Risk':
                    recommendation = "Immediate outreach: Offer $10/mo annual contract commitment credit and 3 months free Tech Support."
                elif risk_level == 'Moderate Risk':
                    recommendation = "Targeted nurture: Promote Auto-Pay $5/mo discount and offer bundled Cloud Backup."
                else:
                    recommendation = "Loyalty rewards: Account is highly stable. Promote premium streaming speed upgrade."

                resp = {
                    'churn_probability': round(prob, 4),
                    'churn_prediction': pred,
                    'risk_level': risk_level,
                    'top_risk_drivers': drivers,
                    'recommendation': recommendation
                }
                self.send_json_response(200, resp)
                return

            except Exception as e:
                self.send_json_response(500, {'error': str(e)})
                return

        # Endpoint: SQL Studio execution
        elif path == '/api/sql':
            query = req_data.get('query', '').strip()
            if not query:
                self.send_json_response(400, {'error': 'Query string cannot be empty'})
                return

            # Safety check: only allow read-only SELECT
            if not query.lower().startswith('select') and not query.lower().startswith('with'):
                self.send_json_response(403, {'error': 'Only SELECT / WITH read-only queries are permitted'})
                return

            t0 = time.time()
            try:
                cursor = SQL_CONN.cursor()
                cursor.execute(query)
                cols = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                elapsed_ms = round((time.time() - t0) * 1000, 2)

                # Format rows for JSON safety
                formatted_rows = []
                for row in rows[:100]:  # limit to top 100 rows for browser performance
                    formatted_rows.append([str(v) if v is not None else 'NULL' for v in row])

                self.send_json_response(200, {
                    'columns': cols,
                    'rows': formatted_rows,
                    'total_rows_returned': len(rows),
                    'execution_time_ms': elapsed_ms
                })
                return

            except Exception as e:
                self.send_json_response(400, {'error': str(e)})
                return

        else:
            self.send_error(404, "Endpoint not found")


def run():
    server_address = ('127.0.0.1', PORT)
    httpd = HTTPServer(server_address, ChurnServerHandler)
    print("="*80)
    print(f"🚀 TELCO CHURN LIVE ANALYTICS & PREDICTION SERVER RUNNING")
    print(f"👉 Local Access URL: http://localhost:{PORT}")
    print(f"👉 Endpoints: / (Dashboard), /api/metrics, /api/predict, /api/sql")
    print("="*80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n--> Shutting down server.")
        httpd.server_close()


if __name__ == '__main__':
    run()
