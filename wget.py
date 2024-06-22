import requests
from pathlib import Path


class Wget:
    @staticmethod
    def get_file(file_path, url):
        if Path(file_path).is_file():
            print(f"{file_path} already exists, skipping download")
        else:
            print(f"Downloading {file_path} ...")
            request = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(request.content)
