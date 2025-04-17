import boto3
from botocore.client import Config

s3 = boto3.client('s3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='72en2S9Xv7qjlnMkx6tj',
    aws_secret_access_key='N5Xpj8P9ftRknAppLQtSwOXT9Wy9ZY2R4J3rMe2s',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

def upload_model(file_path, bucket_name, object_name):
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, object_name)

def download_model(bucket_name, object_name, destination_path):
    with open(destination_path, "wb") as f:
        s3.download_fileobj(bucket_name, object_name, f)
