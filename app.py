from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import shutil
import os
import uuid
from utils import infer_task_type, save_model
from eda import run_eda
from preprocess import preprocess_data
from train_models import train_and_return_models
from evaluate import evaluate_models, print_model_comparison
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure templates directory exists
if not os.path.exists("templates"):
    os.makedirs("templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/train", response_class=HTMLResponse)
def train(request: Request,
          file: UploadFile = File(...),
          target: str = Form(...),
          task: str = Form(None),
          test_size: float = Form(0.2)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = f"uploaded_{file_id}.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = pd.read_csv(file_path)

    # Run EDA (prints to console)
    run_eda(df, target)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target, test_size)
    task_type = task or infer_task_type(df, target)
    models = train_and_return_models(X_train, y_train, task_type)
    results = evaluate_models(models, X_test, y_test, task_type)

    # Find best model by F1 (classification) or R2 (regression)
    if task_type == 'classification':
        best_model_name = max(results, key=lambda k: results[k]['F1'])
    else:
        best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    model_path = f"model_{file_id}.pkl"
    save_model(best_model, model_path)
    # Save preprocessor for future use
    joblib.dump(preprocessor, f"preprocessor_{file_id}.pkl")

    # Prepare metrics table as HTML
    df_results = pd.DataFrame(results).T
    metrics_html = df_results.to_html(classes="table table-bordered", float_format="{:.4f}".format)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "metrics_table": metrics_html,
        "model_path": model_path,
        "best_model_name": best_model_name
    })

@app.get("/download/{model_path}")
def download_model(model_path: str):
    return FileResponse(model_path, media_type='application/octet-stream', filename=model_path) 