import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load CSV data into a DataFrame
df = pd.read_csv('data/results/results.csv')

dataset_path = '/mnt/hdd/datasets/IoTGym/panoptic'

# Set Seaborn style
sns.set_theme(style="white")

def oks_plot(dataset: str, video: str, number_of_id: int, bin_size: int, frames=[2000, 3000]):
    """
    Plot OKS values for a given dataset, video, and idGt.
    """

    filtered_df = df[(df['dataset'] == dataset) & (df['video'] == video) & (df['idGt'] == number_of_id)]
    methods = filtered_df['method'].unique()

    # Create a new DataFrame for plotting
    plot_df = filtered_df[['frame', 'method', 'OKS']].copy()

    # Separate -1 OKS values
    negative_oks_df = plot_df[plot_df['OKS'] == -1]
    # Replace -1 OKS values with NaN
    plot_df['OKS'].replace(-1, np.nan, inplace=True)

    # Aggregate data into bins of specified size
    plot_df.loc[:,'frame_bin'] = (plot_df['frame'] // bin_size) * bin_size
    aggregated_df = plot_df.groupby(['frame_bin', 'method']).agg({'OKS': ['mean', 'std']}).reset_index()
    aggregated_df.columns = ['frame_bin', 'method', 'OKS_mean', 'OKS_std']

    # Compute xlim from aggregated_df
    x_min = aggregated_df['frame_bin'].min()
    x_max = aggregated_df['frame_bin'].max()

    # Create a color palette
    palette = sns.color_palette(None, len(methods))
    color_dict = {method: palette[i] for i, method in enumerate(methods)}

    # Plot OKS values
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Main OKS plot
    for method in methods:
        method_df = aggregated_df[aggregated_df['method'] == method]
        color = color_dict[method]
        axes[0].plot(method_df['frame_bin'], method_df['OKS_mean'], label=method, color=color, alpha=0.6)
        axes[0].fill_between(method_df['frame_bin'], method_df['OKS_mean'] - method_df['OKS_std'], method_df['OKS_mean'] + method_df['OKS_std'], alpha=0.2)

        # add true OKS values for designated frames
        for frame in frames:
           frame_df = plot_df[(plot_df['frame'] == frame) & (plot_df['method'] == method)]              
           axes[0].scatter(frame_df['frame'], frame_df['OKS'], s=100, marker='o', color=color, zorder=5)

    axes[0].set_title(f'Averaged OKS values')
    axes[0].set_ylabel('OKS')
    axes[0].legend()
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(-0.1, 1.1)

    # Remove the box around the first graph
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)

    # Plot negative OKS values for each method on different y values
    for idx, method in enumerate(methods):
        method_df = negative_oks_df[negative_oks_df['method'] == method]
        y_values = np.full(len(method_df), -0.1 - idx * 0.025)
        axes[1].scatter(method_df['frame'], y_values, label=method, s=5, color = color_dict[method], alpha=0.6)

    # Add vertical lines spanning all graphs at frames given by list frames
    for frame in frames:
        axes[0].axvline(x=frame, color='gray', linestyle='--', alpha=0.7, ymin=0, ymax=1)
        axes[1].axvline(x=frame, color='gray', linestyle='--', alpha=0.7)
        # Add frame labels as overlay over the second graph at the bottom of lines
        axes[1].text(frame, -0.1 - len(methods) * 0.1, f'{frame}', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    axes[1].set_title('Non-computed OKS values')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('OKS')
    axes[1].set_ylim(-0.1 - len(methods) * 0.1, 0)
    axes[1].set_xlim(x_min, x_max)
    axes[1].axis('off')  # Turn off axes on axes[1]

    # Set x-ticks on the top subplot
    axes[0].xaxis.set_ticks_position('top')
    axes[0].xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()

    # Create the directory if it doesn't exist
    output_dir = './data/results/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate the filename
    frames_str = '_'.join(map(str, frames))
    filename = f'oks_plot_{dataset}_{video}_{number_of_id}_{bin_size}_{frames_str}.png'
    filepath = os.path.join(output_dir, filename)

    # Save the figure with cropping
    print(f'Saving figure to {filepath}', end='')
    fig.savefig(filepath, bbox_inches='tight')
    print(' - Done!')

def process_data():
    # Iterate over dataset, video, and idGt
    datasets = df['dataset'].unique()
    for d in datasets:
        videos = df[df['dataset'] == d]['video'].unique()
        for v in videos:
            number_of_ids = df[(df['dataset'] == d) & (df['video'] == v) & (df['idGt'] > 0)]['idGt'].unique()
            for i in number_of_ids:
                oks_plot(d, v, i, bin_size=100)

if __name__ == '__main__':
    # Test data
    oks_plot('160906_pizza1', '00_00', 3, 10, frames=[700,800])

    # Uncomment to process all data
    # process_data()