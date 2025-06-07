from utils.preprocessing import load_and_prepare_data
from trainer.train import train_per_city
from utils.utils import save_results

if __name__ == "__main__":
    df = load_and_prepare_data("data/CITYHOURPM25.csv")
    results_df, metrics = train_per_city(df)
    save_results(results_df, metrics)
