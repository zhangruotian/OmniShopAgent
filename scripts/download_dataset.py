"""
Script to download the Fashion Product Images Dataset from Kaggle

Requirements:
1. Install Kaggle CLI: pip install kaggle
2. Setup Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Save kaggle.json to ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_dataset.py
"""

import subprocess
import zipfile
from pathlib import Path


def download_dataset():
    """Download and extract the Fashion Product Images Dataset"""

    # Get project root
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw"

    # Check if data already exists
    if (raw_data_path / "styles.csv").exists():
        print("Dataset already exists in data/raw/")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != "y":
            print("Skipping download.")
            return

    # Check Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(" Kaggle API credentials not found!")
        return

    print("Downloading dataset from Kaggle...")

    try:
        # Download using Kaggle API
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "paramaggarwal/fashion-product-images-dataset",
                "-p",
                str(raw_data_path),
            ],
            check=True,
        )

        print("Download complete!")

        # Extract zip file
        zip_path = raw_data_path / "fashion-product-images-dataset.zip"
        if zip_path.exists():
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_data_path)

            print("Extraction complete!")

            # Clean up zip file
            zip_path.unlink()
            print("Cleaned up zip file")

        # Verify files
        styles_csv = raw_data_path / "styles.csv"
        images_dir = raw_data_path / "images"

        if styles_csv.exists() and images_dir.exists():
            print("\Dataset ready!")

            # Count images
            image_count = len(list(images_dir.glob("*.jpg")))
            print(f"- Total images: {image_count:,}")
        else:
            print("Warning: Expected files not found")

    except subprocess.CalledProcessError:
        print("Download failed!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    download_dataset()
