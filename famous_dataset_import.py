import kagglehub
import shutil
from pathlib import Path


def download_dataset(dataset_name, destination_dir):
    destination_dir = Path(destination_dir)

    if destination_dir.exists():
        print(f"Le dataset existe déjà dans {destination_dir.resolve()}.")
        return destination_dir

    print(f"Téléchargement du dataset '{dataset_name}'...")
    downloaded_path = Path(kagglehub.dataset_download(dataset_name))
    shutil.copytree(downloaded_path, destination_dir)
    print(f"Dataset copié dans {destination_dir.resolve()}.")

    return destination_dir


if __name__ == "__main__":
    download_dataset(
        dataset_name="muhammadhananasghar/5-famous-people-face-recognition",
        destination_dir="famous_people_dataset"
    )