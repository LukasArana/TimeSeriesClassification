import os
import pandas as pd
import numpy as np
from project3 import classifiers, datasets, datasets_small
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import glob
classifiers_names = list(classifiers.keys())
results_path = "results"
time_path = "results/time.csv"
mpl.rc('font', family='Serif', size=12)

sns.set_style("whitegrid")

#convert all to a function
def get_data():
    accs = np.zeros((len(classifiers_names), len(datasets)) )
    for name_csv in glob.glob(os.path.join(results_path, "*.csv")):
        if "time.csv" in name_csv:
            continue
        #open name_csv with pandas
        df = pd.read_csv(name_csv)
        name_csv = name_csv.split("/")[-1]
        #get name of algorithm from name_csv
        dataset_name = name_csv.split("_")[0]
        alg_name = "_".join(name_csv.split("_")[1:])[0:-4]
        #read cols of the csv
        preds = df["pred"]
        true = df["true"]
        #calculate accuracy between preds and true
        acc = accuracy_score(true, preds)
        #conf_matrix = confusion_matrix(true, preds)

        #save accuracy in accs
        accs[list(classifiers_names).index(alg_name), list(datasets).index(dataset_name)] = acc
    return accs

def violinPlot(accs, path, show = True):
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Change the background color of the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Select datasets where none of the classifiers have 0 as accuracy
    zero_acc_datasets = np.all(accs != 0, axis=0)
    a = np.any(accs == 0, axis=0)
    print(datasets[a])
    accs_filtered = accs[:, zero_acc_datasets]

    # Create a DataFrame from the filtered accuracies
    df = pd.DataFrame(accs_filtered.T, columns=classifiers_names)

    # Calculate the median of each classifier's accuracy
    medians = df.median().sort_values()

    # Reorder the DataFrame columns by the median
    df = df[medians.index]

    # Create violin plot for each classifier's accuracy on each dataset
    sns.violinplot(data=df, palette="Set3", inner="quartile", ax=ax)
    # Create custom legend
    handles = [mlines.Line2D([], [], color='black', linestyle='dotted', markersize=15, label='25th Quartile'),
            mlines.Line2D([], [], color='black',linestyle="--", linewidth=1, markersize=15, label='Median'),
            mlines.Line2D([], [], color='black', linestyle='dotted', markersize=15, label='75th Quartile')]
    ax.legend(handles=handles, loc='lower right')# Set title and labels with more details
    #ax.set_title('Distribution of Accuracy for each Classifier across Datasets', fontsize=16)
    ax.set_xlabel('Classifier', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1)

    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    fig.savefig(path, dpi = 300)
    if show:
        plt.show()
    plt.close()

#Plot barplot of accuracy for each dataset
def barPlot(accs, path, show = True):

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Change the background color of the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Select classifiers where none of the datasets have 0 as accuracy
    zero_acc_classifiers = np.all(accs != 0, axis=1)
    accs_filtered = accs[zero_acc_classifiers]

    # Create a DataFrame from the filtered accuracies
    df = pd.DataFrame(accs_filtered, columns=datasets)

    # Calculate the median of each dataset's accuracy
    medians = df.median().sort_values()

    # Reorder the DataFrame columns by the median
    df = df[medians.index]

    # Create bar plot for each dataset's accuracy on each classifier
    fig, ax = plt.subplots(figsize=(10, 6))

    # Change the background color of the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Select classifiers where none of the datasets have 0 as accuracy
    zero_acc_classifiers = np.all(accs != 0, axis=1)
    accs_filtered = accs[zero_acc_classifiers]

    # Create a DataFrame from the filtered accuracies
    df = pd.DataFrame(accs_filtered, columns=datasets_small)
    # Calculate the median of each dataset's accuracy
    medians = df.median().sort_values()

    # Reorder the DataFrame columns by the median
    df = df[medians.index]

    # Create bar plot for each dataset's accuracy on each classifier
    sns.barplot(data=df, palette="Set3", ax=ax, edgecolor='none', errorbar=None)

    #ax.set_title('Distribution of Accuracy for each Dataset across Classifiers', fontsize=16)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    accs = get_data()
    violinPlot(accs, "results/images/violin_plot.png", show = False)
    barPlot(accs, "results/images/bar_plot.png", show  =False)
    