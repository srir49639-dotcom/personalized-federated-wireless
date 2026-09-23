# Application Architecture: Personalized Federated Wireless Link Quality Predictor

This directory contains the production web application and inference API for the **Personalized Federated Deep Learning for Wireless Link Quality Prediction** project.

## Directory Structure

```
app/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, route mounting & static serving
│   ├── model_loader.py      # Architecture reconstruction & cached client model loading
│   ├── preprocessing.py     # 472-feature pipeline (465 WAPs + 7 engineered statistics)
│   ├── predictor.py         # Locked 0.25 dBm margin hybrid predictor & fallback logic
│   ├── analytics.py         # Precomputed artifact metrics, confusion matrix, non-IID tables
│   └── schemas.py           # Pydantic request/response validation schemas
│
├── frontend/
│   ├── index.html           # Full research dashboard single-page application
│   ├── css/
│   │   └── styles.css       # Complete dark/light design system with responsive layout
│   ├── js/
│   │   ├── app.js           # Navigation, data fetchers, theme toggle
│   │   ├── charts.js        # Interactive Chart.js visualizations
│   │   └── predictor.js     # Live prediction controller & RSSI gauge
│   └── assets/
│       └── logo.svg         # SVG vector logo
│
└── app.py                   # Preserved legacy Gradio application
```

## Running the Web Application

To run the modern FastAPI dashboard, from the repository root run:

```bash
python run.py
```

Then open `http://127.0.0.1:8000` in your web browser.

Interactive Swagger API documentation is available at:
`http://127.0.0.1:8000/docs`
