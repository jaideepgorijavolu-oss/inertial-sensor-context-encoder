"""Download and extract the UCI HAR dataset into data/UCI HAR Dataset."""
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def extract_all(zf: zipfile.ZipFile, dest: str):
    zf.extractall(dest)
    # The UCI archive wraps the real dataset in a second zip.
    for name in zf.namelist():
        if name.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(dest, name)) as inner:
                extract_all(inner, dest)


def main(dest: str = "data"):
    target = os.path.join(dest, "UCI HAR Dataset")
    if os.path.isdir(os.path.join(target, "train", "Inertial Signals")):
        print(f"Dataset already present at {target}")
        return
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        extract_all(zf, dest)
    if not os.path.isdir(os.path.join(target, "train", "Inertial Signals")):
        sys.exit(f"Extraction finished but {target}/train/Inertial Signals is missing")
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
