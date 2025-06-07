import matplotlib.pyplot as plt
import os

def plot_anomalies(df, City, save_dir='results/City_wise_plots'):
    os.makedirs(save_dir, exist_ok=True)
    df_city = df[df['City'] == City]
    plt.figure(figsize=(14, 5))
    plt.plot(df_city['Datetime'], df_city['PM2.5'], label='PM2.5')
    plt.scatter(df_city[df_city['anomaly']]['Datetime'], df_city[df_city['anomaly']]['PM2.5'], color='red', label='Anomaly', s=15)
    plt.title(f"Anomaly Detection - {City}")
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{City}_anomalies.png")
    plt.close()
