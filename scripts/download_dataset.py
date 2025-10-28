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

    print("=" * 60)
    print("Fashion Product Images Dataset Downloader")
    print("=" * 60)

    # Check if data already exists
    if (raw_data_path / "styles.csv").exists():
        print("\n✓ Dataset already exists in data/raw/")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != "y":
            print("Skipping download.")
            return

    # Check Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n❌ Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return

    print("\n📥 Downloading dataset from Kaggle...")
    print("Dataset: fashion-product-images-dataset (~15 GB)")

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

        print("\n✓ Download complete!")

        # Extract zip file
        zip_path = raw_data_path / "fashion-product-images-dataset.zip"
        if zip_path.exists():
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_data_path)

            print("✓ Extraction complete!")

            # Clean up zip file
            zip_path.unlink()
            print("✓ Cleaned up zip file")

        # Verify files
        styles_csv = raw_data_path / "styles.csv"
        images_dir = raw_data_path / "images"

        if styles_csv.exists() and images_dir.exists():
            print("\n✅ Dataset ready!")
            print(f"   - Metadata: {styles_csv}")
            print(f"   - Images: {images_dir}")

            # Count images
            image_count = len(list(images_dir.glob("*.jpg")))
            print(f"   - Total images: {image_count:,}")
        else:
            print("\n⚠️  Warning: Expected files not found")
            print(f"   Looking for: {styles_csv}")
            print(f"   and: {images_dir}")

    except subprocess.CalledProcessError:
        print("\n❌ Download failed!")
        print("Make sure you have:")
        print("1. Installed Kaggle CLI: pip install kaggle")
        print("2. Setup API credentials in ~/.kaggle/kaggle.json")
        print("3. Accepted the dataset terms on Kaggle website")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    download_dataset()
