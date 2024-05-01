import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('../document/AnnualReports16_position.csv', header=0)
dataset = dataset[dataset['position'] != "[]"]

# Sample data
data = dataset['position'].apply(ast.literal_eval).to_list()
full_position = []
avg_position = []
for position_set in data:
    if position_set[-1] > 50:
        position_list = [item / position_set[-1] for item in position_set[:-1]]
        avg_position.append(sum(position_list) / len(position_list))
        full_position.extend(position_list)

sns.histplot(full_position, kde=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Full Distribution Plot')
plt.show()

sns.histplot(avg_position, kde=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Avg Distribution Plot')
plt.show()
