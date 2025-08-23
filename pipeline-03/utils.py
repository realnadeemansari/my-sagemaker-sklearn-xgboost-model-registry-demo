import subprocess
import os
import boto3
import requests

def start_mlflow_ui():
    subprocess.Popen(
        ["mlflow", "ui", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def upload_to_s3(local_dir, bucket, prefix):
    s3 = boto3.client("s3") 
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(prefix, os.path.relpath(local_path, local_dir))
            s3.upload_file(local_path, bucket, s3_path)


def download_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_path = obj["Key"]
            local_path = os.path.join(local_dir, os.path.relpath(s3_path, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_path, local_path)

def download_cacert(save_path="./cacert.pem"):
    url = "https://curl.se/ca/cacert.pem"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(r.content)

    print(f"CA certificate saved to {save_path}")