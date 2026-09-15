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
