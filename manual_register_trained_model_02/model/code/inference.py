import os
import joblib
import pandas as pd
import io


def model_fn(model_dir):
    """Load and return the trained model."""
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def input_fn(input_data, content_type):
    """Deserialize input data from the request."""
    if content_type == "text/csv":
        # Convert CSV string to Pandas DataFrame
        columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        df = pd.read_csv(io.StringIO(input_data), header=None)
        df.columns = columns
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run prediction on the deserialized input."""
    return model.transform(input_data)


def output_fn(prediction, accept):
    """Serialize prediction output."""
    if accept == "text/csv":
        return ",".join(str(x) for x in prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
