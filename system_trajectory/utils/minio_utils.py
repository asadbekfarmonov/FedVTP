from minio import Minio
from minio.error import S3Error
import os
# mc alias set local http://localhost:9000 72en2S9Xv7qjlnMkx6tj N5Xpj8P9ftRknAppLQtSwOXT9Wy9ZY2R4J3rMe2s
endpoint = os.environ.get("MINIO_URL", "minio.default.svc.cluster.local:9000")
access_key = os.environ.get("MINIO_ACCESS_KEY", "72en2S9Xv7qjlnMkx6tj")
secret_key = os.environ.get("MINIO_SECRET_KEY", "N5Xpj8P9ftRknAppLQtSwOXT9Wy9ZY2R4J3rMe2s")

print(f"[MinIO INIT] Connecting to MinIO at {endpoint}")
print(f"[MinIO INIT] Access Key: {access_key[:4]}***")  # Don't log full secrets

client = Minio(
    endpoint=endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

def ensure_bucket_exists(bucket_name):
    try:
        exists = client.bucket_exists(bucket_name)
        print(f"[MinIO] Bucket '{bucket_name}' exists: {exists}")
        if not exists:
            print(f"[MinIO] Creating bucket: {bucket_name}")
            client.make_bucket(bucket_name)
        else:
            print(f"[MinIO] Bucket '{bucket_name}' already exists")
    except S3Error as e:
        print(f"[MinIO ERROR] Checking bucket existence failed: {e}")
        raise

def upload_model(file_path, bucket_name, object_name):
    try:
        ensure_bucket_exists(bucket_name)
        print(f"[UPLOAD MODEL] Uploading '{file_path}' to bucket '{bucket_name}' as '{object_name}'")
        client.fput_object(bucket_name, object_name, file_path)
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        raise

def download_model(bucket_name, object_name, destination_path):
    print(f"[DOWNLOAD MODEL] Downloading '{object_name}' from bucket '{bucket_name}' to '{destination_path}'")
    client.fget_object(bucket_name, object_name, destination_path)

def list_models(bucket_name, prefix=""):
    print(f"[LIST MODELS] Listing from bucket '{bucket_name}' with prefix '{prefix}'")
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]

def upload_generic_file(bucket_name, object_name, file_path):
    ensure_bucket_exists(bucket_name)
    print(f"[UPLOAD FILE] Uploading '{file_path}' to '{bucket_name}/{object_name}'")
    client.fput_object(bucket_name, object_name, file_path)

def download_file(bucket_name, object_name, file_path):
    print(f"[DOWNLOAD FILE] Downloading '{object_name}' from bucket '{bucket_name}' to '{file_path}'")
    client.fget_object(bucket_name, object_name, file_path)

def upload_file(bucket_name: str, object_name: str, file_path: str):
    try:
        client.fput_object(bucket_name, object_name, file_path)
        print(f"[MinIO] Uploaded file to {bucket_name}/{object_name}")
    except S3Error as e:
        print(f"[MinIO ERROR] {e}")