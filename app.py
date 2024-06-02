from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import os

from keras import *
from keras.src.saving import load_model

from src.tonal_project.pipelines.model_prediction.nodes import predict_model

app = Flask(__name__)
metadata = bootstrap_project(Path.cwd())

project_path = Path("tonal-project")

# Load the model
model_path = "data/06_models/model_trained.h5"  # Update this path to the location of your model
model = load_model(model_path)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload_predict', methods=['GET'])
def upload_predict():
    return render_template('upload_predict.html')


@app.route('/upload_aggregate', methods=['GET'])
def upload_aggregate():
    return render_template('upload_aggregate.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the uploaded file
        file = request.files['file']
        raw_data_df = pd.read_csv(file)

        # Use the predict_model function to get predictions
        predictions = predict_model(raw_data_df, model)

        # Combine the input data with predictions for display
        data_to_predict = pd.concat([raw_data_df, predictions], axis=1)

        return render_template('results.html', tables=[data_to_predict.to_html(classes='data')],
                               titles=data_to_predict.columns.values)
    except Exception as e:
        return redirect(url_for('home', error=str(e)))

@app.route('/aggregate', methods=['GET', 'POST'])
def aggregate():
    if request.method == 'POST':
        try:
            file = request.files['file']
            aggregate_data_df = pd.read_csv(file)

            tonal_exams_path = "data/01_raw/tonal_exams.csv"

            if os.path.exists(tonal_exams_path):
                existing_data_df = pd.read_csv(tonal_exams_path)
            else:
                existing_data_df = pd.DataFrame()

            combined_df = pd.concat([existing_data_df, aggregate_data_df], ignore_index=True)

            combined_df.to_csv(tonal_exams_path, index=False)

            with KedroSession.create(project_path=".") as session:
                session.run(pipeline_name="__default__")

            return redirect(url_for('home', success="Model training completed successfully!"))
        except Exception as e:
            return redirect(url_for('home', error=str(e)))
    return render_template('aggregate.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002, debug=True)
