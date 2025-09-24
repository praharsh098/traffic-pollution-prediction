import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_traffic_pollution(df):
    plt.figure(figsize=(15,5))
    plt.plot(df['datetime'], df['traffic_volume'], label='Traffic Volume')
    plt.plot(df['datetime'], df['pm25'], label='PM2.5')
    plt.xlabel('Datetime')
    plt.ylabel('Values')
    plt.title('Traffic & Pollution Trends')
    plt.legend()
    plt.tight_layout()

    # Resolve save path to project root outputs/plots
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'traffic_pollution_trends.png')

    plt.savefig(out_path)
    plt.show()
