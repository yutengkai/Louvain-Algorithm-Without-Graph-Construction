import requests
import zipfile
import os
import shutil

def download_and_extract_zip(url, extract_to):
    """
    Download a zip file from a URL and extract its contents.

    Parameters:
    - url (str): The URL to download the zip file from.
    - extract_to (str): The directory to extract the contents to.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the zip file
    zip_path = os.path.join(extract_to, "temp.zip")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Remove the zip file after extraction
    os.remove(zip_path)

    # Move the JSON files to the specified directory
    json_files = [file for file in os.listdir(extract_to) if file.endswith('.json')]
    for json_file in json_files:
        shutil.move(os.path.join(extract_to, json_file), os.path.join(extract_to, "movie_dataset_public_final/raw", json_file))

def download_ml_25m():
	url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
	extract_to = "."
	download_and_extract_zip(url, extract_to)
	print("Download and extraction complete.")