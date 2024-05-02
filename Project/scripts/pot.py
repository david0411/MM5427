import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('../document/16_word_position.csv', header=0)
dataset = dataset[dataset['position'] != "[]"]

data = dataset['position'].apply(ast.literal_eval).to_list()
full_position = []
avg_position = []
for position_set in data:
    if position_set[-1] > 50:
        position_list = [item / position_set[-1] for item in position_set[:-1]]
        avg_position.append(sum(position_list) / len(position_list))
        full_position.extend(position_list)
    else:
        avg_position.append()

# sns.histplot(avg_position, kde=True)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Avg Distribution Plot')
# plt.show()

temp_df = pd.DataFrame(dataset['item7']).copy()
temp_df['word_position_index'] = avg_position
