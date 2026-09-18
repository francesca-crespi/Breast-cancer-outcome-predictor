def clean_breast_cancer_data(df):
    """Clean and encode the breast cancer dataset."""
    columns_to_drop = ["id", "unnamed: 32"]
    df_clean = df.drop(
        columns=[column for column in columns_to_drop if column in df.columns]
    )

    missing_values = df_clean.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    print("Missing values per column:")
    print(missing_values)

    # Malignant = 1, Benign = 0
    df_clean["diagnosis"] = df_clean["diagnosis"].map({"M": 1, "B": 0})

    print(
        "Unique values in 'diagnosis' after encoding:",
        df_clean["diagnosis"].unique(),
    )

    num_duplicates = df_clean.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates}")

    df_clean = df_clean.drop_duplicates()

    print(f"Shape of dataset after cleaning: {df_clean.shape}")

    return df_clean
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    
    # Get the root project directory dynamically based on script location
    script_dir = Path(__file__).resolve().parent
    ruta_dataset = script_dir.parent.parent / "data" / "data.zip"
    
    print("--- Cargando los datos de Cáncer de Mama ---")
    
    try:
        # Leemos el archivo utilizando pandas (soporta archivos .zip directamente)
        df_original = pd.read_csv(ruta_dataset)
        print("¡Dataset cargado con éxito!")
        print("-" * 40)
        
        # Llamamos a tu función de limpieza
        clean_breast_cancer_data(df_original)
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta '{ruta_dataset}'.")
        print("Revisa si tu archivo se llama 'data.csv', 'data.txt' o si está en otra carpeta.")