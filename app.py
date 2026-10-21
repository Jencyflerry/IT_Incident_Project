from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --------------------------------------------------
# 1. Load trained model and metadata
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run Incident_code.ipynb first."
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}. "
        "Run Incident_code.ipynb first."
    )

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

priority_mapping = metadata["priority_mapping"]

reverse_mapping = {
    value: key for key, value in priority_mapping.items()
}


# --------------------------------------------------
# 2. Home page
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# --------------------------------------------------
# 3. Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Direct browser access should show the input form.
    if request.method == "GET":
        return render_template("index.html")

    try:
        # Read submitted form values
        data = {
            "Category": request.form["Category"],
            "Subcategory": request.form["Subcategory"],
            "Impact": request.form["Impact"],
            "Urgency": request.form["Urgency"],
            "Affected_Users": int(
                request.form["Affected_Users"]
            ),
            "Service_Criticality": request.form[
                "Service_Criticality"
            ],
            "Business_Hours": request.form["Business_Hours"],
            "Customer_Impact": request.form["Customer_Impact"],
            "Reopen_Count": int(
                request.form["Reopen_Count"]
            ),
            "Assignment_Group": request.form[
                "Assignment_Group"
            ],
        }

        # Basic numeric validation
        if not 0 <= data["Affected_Users"] <= 5000:
            raise ValueError(
                "Affected Users must be between 0 and 5000."
            )

        if data["Reopen_Count"] < 0:
            raise ValueError(
                "Reopen Count cannot be negative."
            )

        # Create input DataFrame in training feature order
        df = pd.DataFrame([data])
        df = df[metadata["features"]]

        # Predict incident priority
        prediction = int(model.predict(df)[0])

        predicted_priority = reverse_mapping[prediction]

        # Calculate probabilities when supported by the model
        priority_probabilities = {}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[0]

            if hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")
            else:
                classifier = model

            if classifier is not None and hasattr(
                classifier, "classes_"
            ):
                classes = classifier.classes_
            elif hasattr(model, "classes_"):
                classes = model.classes_
            else:
                classes = []

            for cls, probability in zip(
                classes, probabilities
            ):
                label = reverse_mapping.get(
                    int(cls), str(cls)
                )

                priority_probabilities[str(label)] = round(
                    float(probability) * 100, 2
                )

        # Render result page with ALL required variables
        return render_template(
            "result.html",
            priority=predicted_priority,
            probabilities=priority_probabilities,
            incident=data,
            model_name=metadata.get(
                "best_model_name", "Trained ML Model"
            ),
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            error=str(error)
        ), 400

    except Exception:
        app.logger.exception("Incident prediction failed")

        return (
            "Prediction failed. Check the VS Code terminal "
            "for the detailed error.",
            500,
        )


# --------------------------------------------------
# 4. Start Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
