# 📊 RetailPulse AI Analytics

## AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is an end-to-end retail analytics platform that uses machine learning and business intelligence techniques to analyze customer behavior, forecast future demand, predict customer churn, and optimize inventory decisions for retail businesses.

The project was developed as part of the Zidio Data Science & Analytics domain project with a strong focus on real-world retail analytics, predictive modeling, and interactive business dashboards.

---

## 🚀 Live Demo
https://retailpulse-ai-analytics-bcpoeovsdpq3yh7nkerfmd.streamlit.app/

---

# 🚀 Project Objectives

- Analyze retail sales and customer behavior
- Segment customers using RFM analysis and clustering
- Forecast future sales demand using time-series forecasting
- Predict customer churn risk using machine learning
- Optimize inventory and reorder recommendations
- Build an interactive analytics dashboard using Streamlit

---

# ✨ Key Features

## 🔹 Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis
- KMeans clustering
- VIP customer identification
- Customer behavior analysis

## 🔹 Demand Forecasting
- Prophet forecasting model
- LSTM forecasting model
- Ensemble forecasting comparison
- 30-day future revenue prediction

## 🔹 Churn Prediction
- Random Forest churn prediction
- XGBoost model comparison
- ROC-AUC evaluation
- Precision@Top20% evaluation

## 🔹 Inventory Optimization
- ABC inventory analysis
- Demand classification
- Reorder recommendation system
- Estimated reorder quantity generation

## 🔹 Interactive Dashboard
- Streamlit multi-page dashboard
- KPI metrics
- Country-based filtering
- Downloadable CSV reports
- Interactive business insights

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Forecasting | Prophet, TensorFlow LSTM |
| Dashboard | Streamlit |
| Development Tools | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

# 📁 Project Structure

```text
RetailPulse-AI-Analytics/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── rfm_segmentation.ipynb
│   ├── demand_forecasting.ipynb
│   ├── churn_prediction.ipynb
│   └── inventory_optimization.ipynb
│
├── outputs/
│   ├── charts/
│   ├── forecast_results/
│   ├── churn_prediction_results.csv
│   ├── inventory_optimization_results.csv
│   ├── rfm_customer_segments.csv
│   └── model_metrics.csv
│
├── screenshots/


```
---

# 📈 Model Performance
| Module                | Model            | Performance         |
| --------------------- | ---------------- | ------------------- |
| Demand Forecasting    | Prophet          | MAPE ≈ 22.66%       |
| Churn Prediction      | Random Forest    | ROC-AUC ≈ 0.82      |
| Churn Prediction      | Precision@Top20% | ≈ 0.72              |
| Customer Segmentation | KMeans           | 6 customer segments |

---

# 💡 Business Insights
High-value VIP customers were identified using RFM analysis
Prophet forecasting captured seasonal retail demand trends effectively
Customer engagement duration was a strong churn indicator
Inventory optimization helped identify high-priority restock products
ABC analysis highlighted products contributing the highest revenue

---

# 🖥️ Streamlit Dashboard

The Streamlit dashboard provides:

Business KPI overview
Sales analytics
Customer segmentation insights
Demand forecasting visualizations
Churn prediction analysis
Inventory optimization recommendations

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/RetailPulse-AI-Analytics.git
```

## 2️⃣ Move Into Project Folder

```bash
cd RetailPulse-AI-Analytics
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

## 5️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```


---

# 🔮 Future Improvements
Hyperparameter optimization using Optuna
Drift detection using Evidently AI
MLflow experiment tracking
Docker containerization
Cloud deployment on AWS/GCP
Real-time forecasting pipeline

---

# ⚠️ Challenges Faced
Handling highly volatile retail demand patterns
Managing seasonal sales spikes and holiday effects
Avoiding data leakage in churn prediction
Balancing model accuracy and interpretability

---

# 👨‍💻 Author
Indra Jaiswal

---

# 📜 License

This project is developed for educational and portfolio purposes under the Zidio Data Science & Analytics program.

---














