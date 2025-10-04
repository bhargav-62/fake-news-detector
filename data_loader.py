import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path, test_size=0.2, random_state=42):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Drop missing or blank entries (if any)
    df = df.dropna(subset=["text", "label"])
    # Optional: Double check column types and one-hot-encode if necessary
    if df["label"].dtype != object:
        df["label"] = df["label"].map({0: "FAKE", 1: "REAL"})
    # Show distribution for debug
    print("Label distribution:\n", df["label"].value_counts())
    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return train_df, val_df

# Test loading for debugging
if __name__ == "__main__":
    train, val = load_data("news_dataset.csv")
    print("Train samples:", len(train))
    print("Validation samples:", len(val))
    print("\nTrain sample rows:\n", train.head())
