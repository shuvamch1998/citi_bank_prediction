# CitiBike Hourly Demand Forecasting System 🚲📈

This project builds a full-stack MLOps pipeline to forecast hourly demand for CitiBike rides across New York City stations. It integrates data engineering, machine learning, model tracking, versioned forecasts, and interactive dashboards.

---

## 🔧 Tech Stack
- **Python, Pandas, LightGBM**
- **AWS SageMaker Feature Store** (for training features)
- **AWS S3 + Athena** (for versioned batch features and forecasts)
- **MLflow + DagsHub** (experiment tracking and model registry)
- **Streamlit** (interactive forecast visualization)

---

## 📦 Workflow Overview
1. **Data Ingestion**: Downloads and filters monthly CitiBike ride data.
2. **Feature Engineering**: Creates time-series lag features and time-based attributes.
3. **Feature Storage**:
   - Stored in SageMaker Feature Store (for lineage)
   - Migrated to S3 via Athena for cheaper querying and analytics
4. **Model Training**:
   - Baseline model (lag average)
   - LightGBM with full and reduced features
   - Best MAE: ~1.59 (LightGBM Full)
5. **Inference**: Recursive batch forecasting over next 90 days.
6. **Forecast Storage**: Written to S3 (`latest.csv` + timestamped).
7. **Dashboards**:
   - `citibike_monitor.py`: Exploratory view with calendar filters
   - `citibike_dashboard.py`: Real-time operational view

---

## 📊 MAE Results
| Model                  | MAE   |
|------------------------|-------|
| BaselinePredictor      | ~4.25 |
| LightGBM (Full)        | ~1.59 |
| LightGBM (Reduced)     | ~1.90 |

---

## ✅ Highlights
- Efficient hybrid storage (SageMaker FS for training, S3 for inference)
- End-to-end model lifecycle management via MLflow + DagsHub
- Fully automated, reproducible, and visualized

---

## 📈 Dashboards

- Access forecasts and station trends in real time.
- Export forecasts to CSV.
- Detect daily/weekly patterns interactively.
  
  ![image](https://github.com/user-attachments/assets/02173ea4-68ec-4dfc-943a-019587e2e38b)
  ![image](https://github.com/user-attachments/assets/191c6609-b201-4fbe-9390-1f176b0ded5d)



---
