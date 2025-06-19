import os
import requests

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {dest_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def download_gdrive_file(file_id, dest_path):
    # Handles large files from Google Drive
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {dest_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # Prototxt files (GitHub raw links)
    age_proto_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
    gender_proto_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
    download_file(age_proto_url, "models/age_deploy.prototxt")
    download_file(gender_proto_url, "models/gender_deploy.prototxt")

    # Caffemodel files (Google Drive file IDs)
    age_caffemodel_id = "0B1tW_VtLh2wQWmRoc1VnQURfd2M"
    gender_caffemodel_id = "0B1tW_VtLh2wQbWl1Q1dUQkZFdDQ"
    download_gdrive_file(age_caffemodel_id, "models/age_net.caffemodel")
    download_gdrive_file(gender_caffemodel_id, "models/gender_net.caffemodel") 