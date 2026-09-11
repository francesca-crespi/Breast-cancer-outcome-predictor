from pathlib import Path
import zipfile
import pandas as pd


def load_breast_cancer(data_zip_path, extract_dir="data_extracted"):
    """
    Extract and load the Breast Cancer Wisconsin dataset.

    Parameters
    ----------
    data_zip_path : str or Path
        Path to the ZIP file containing the dataset.
    extract_dir : str or Path
        Directory where the ZIP file will be extracted.

    Returns
    -------
    pandas.DataFrame
        Breast cancer dataset.
    """

    data_zip_path = Path(data_zip_path)
    extract_dir = Path(extract_dir)

    # Create extraction directory if it does not exist
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find CSV files inside the extracted folder
    csv_files = list(extract_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file was found in the extracted dataset."
        )

    # Load the first CSV file found
    data = pd.read_csv(csv_files[0])

    print(f"Dataset loaded from: {csv_files[0]}")
    print(f"Dataset shape: {data.shape}")

    return data
