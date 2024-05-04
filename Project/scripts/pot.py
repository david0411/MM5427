import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('../document/18_combine.csv', header=0)

sns.histplot(dataset['avg_tense_position'], kde=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('ATP Distributions')
plt.show()

sns.histplot(dataset['avg_word_position'], kde=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('AWP Distributions')
plt.show()
