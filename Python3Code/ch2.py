from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys
import pandas as pd

dataset = pd.read_csv(r'C:\Users\moham\Desktop\ML4QS\gestures.csv')

# Plot the data
DataViz = VisualizeDataset(__file__)

# Boxplot
DataViz.plot_dataset_boxplot(dataset, ['gfc_x','gfc_y','gfc_z','acc_x','acc_y','acc_z',
                                       'gyr_x','gyr_y','gyr_z'])#,'inc_x','inc_y','inc_z',
                                    #    'mag_x','mag_y','mag_z'])

# # Plot all data
# DataViz.plot_dataset(dataset, ['gfc_x','gfc_y','gfc_z','acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z' ,'inc_x','inc_y','inc_z','label'],
#                                 ['like','like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like', 'like', 'like', 'like','like'],
#                                 ['line','line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line','line','line', 'points', 'points'])

DataViz.plot_dataset(dataset, ['acc_','gfc_','gyc_','inc_','mag_', 'label'],
                                ['like', 'like', 'like', 'like', 'like', 'like','like'],
                                ['line', 'line', 'line', 'line', 'line', 'points', 'points'])

# And print a summary of the dataset.
util.print_statistics(dataset)
# datasets.append(copy.deepcopy(dataset))