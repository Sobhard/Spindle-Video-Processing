import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tracking_data = pd.read_csv("data_analysis/results.csv")
print(tracking_data[:4])

plt.figure()
sns.lineplot(data=tracking_data, x="timestamp", y="red_x1")

plt.show()
