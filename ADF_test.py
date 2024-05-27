import numpy as np
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
import pandas as pd

plot_params = {'legend.fontsize': 26,
               'figure.figsize': (16, 9),
               'xtick.labelsize': '18',
               'ytick.labelsize': '18',
               'axes.titlesize': '24',
               'axes.labelsize': '22'}

plt.rcParams.update(plot_params)
adf_seq = pd.read_csv(r'data\body-sway\S4\ECL2.csv' , header=0)
adf_seq = adf_seq.iloc[1:10000, 2]
dftest = adfuller(adf_seq,autolag='AIC')
plt.plot(range(len(adf_seq)),adf_seq)
dfoutput = pd.Series(dftest[4],index=['p-value'])
print(dftest)
plt.show()