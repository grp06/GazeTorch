import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde 

'''
Joe: 
- Takes in a list of datasets, labels -> plots graphs that compare the different datasets.
- Mostly used for plotting distributions of pitch/yaw of the headposes/gaze angles.
- For specific example usage, take a look at /data/zerogaze/plot_comparisons for plotting the pitch and yaw distribution for L1 diff across 3 datasets.

- simple example of plotting two datasets' pitch/yaw: 

>> from tools.plotter import Plotter

>> data1 = np.load('/path/to/data1.npy', allow_pickle=True)
>> data2 = np.load('/path/to/data2.npy', allow_pickle=True)

# define dataset as a list of data(npy or list format)
>> datasets = [data1, data2]

# define corresponding labels for the data in the datasets
>> labels = ['data1', 'data2']

#if degrees=True, Plotter converts data to degrees; make sure data has same format
>> plotter = Plotter(datasets=datasets, labels=labels, degrees=True) 

>> plotter.plot_comparison(title="Comparing P/Y of data1 v.s data2", save_path="/path/to/save/")
'''


class Plotter():
    def __init__(self, datasets: list, labels: list[str], feature_names=['Pitch', 'Yaw'], degrees=True):
        """
        datasets: List of np.arrays, each shape (N, 2) for [Pitch, Yaw]
        labels: List of strings (e.g., ['base', 'mirrored'])
        feature_names: List of names for the columns in the data
        """
        self.labels = labels
        self.feature_names = feature_names
        
        # Convert all datasets to degrees if requested
        if degrees:
            self.data_lst = [np.degrees(d) for d in datasets]
        else:
            self.data_lst = datasets

        self.colors = ['skyblue', 'salmon', 'lime', 'orange', 'purple']

    def plot_comparison(self, 
                        bins=150, 
                        range_val=(-30, 30), 
                        title="",
                        add_centerline=True, 
                        use_kde=True,
                        save_path=""):
        
        num_features = len(self.feature_names)
        fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 5))

        if num_features == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
                    for d_idx, data in enumerate(self.data_lst):
                        feature_data = data[:, i]
                        # Remove NaNs to avoid KDE errors
                        feature_data = feature_data[~np.isnan(feature_data)]
                        
                        if use_kde:
                            # Create smooth Kernel Density Estimate
                            kde = gaussian_kde(feature_data)
                            x_eval = np.linspace(range_val[0], range_val[1], 500)
                            ax.plot(x_eval, kde(x_eval), 
                                    color=self.colors[d_idx % len(self.colors)], 
                                    lw=2, label=self.labels[d_idx])
                            # Optional: Fill area under curve
                            ax.fill_between(x_eval, kde(x_eval), alpha=0.2, 
                                            color=self.colors[d_idx % len(self.colors)])
                        else:
                            # Normal Histogram but with density=True
                            ax.hist(feature_data, 
                                    bins=bins, 
                                    range=range_val, 
                                    density=True, # This makes it a density plot
                                    color=self.colors[d_idx % len(self.colors)],
                                    alpha=0.5, 
                                    label=self.labels[d_idx],
                                    histtype='stepfilled') # Cleaner look for density

                    ax.set_title(f'{self.feature_names[i]} Density ({title})')
                    ax.set_xlabel(f'{self.feature_names[i]} (°)')
                    ax.set_ylabel("Density")
                    ax.set_xlim(range_val)
                    ax.grid(axis='y', alpha=0.3)
                    
                    if add_centerline:
                        ax.axvline(0, color='black', linestyle='dashed', linewidth=0.8)
                    
                    ax.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()