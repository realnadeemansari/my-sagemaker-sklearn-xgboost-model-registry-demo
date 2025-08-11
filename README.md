# my-sagemaker-sklearn-xgboost-model-registry-demo
# SageMaker Scikit-learn Model Registry Demo

This repository contains a step-by-step example of:
1. Packaging a local Scikit-learn model.
2. Registering the model into the **Amazon SageMaker Model Registry**.
3. Approving the registered model for deployment.
4. Deploying the model to a real-time SageMaker endpoint.
5. Making predictions using the deployed endpoint.

---

## 📂 Project Structure

.
├── manual_register_trained_model_02.ipynb # Main notebook
├── model/ # Trained model artifacts
│ ├── model.joblib
│ └── code/
│ └── inference.py # Custom inference script
└── README.md