import numpy as np
from collections import Counter
import re
import numpy as np
from collections import Counter
import re

class DatasetAnalysis:
    """."""

    def __call__(self):
        """."""
        data = np.load('image_data.npy', allow_pickle=True)
        word_counts = Counter()

        for d in data:
            text = d['anchor']
            words = re.findall(r'\w+', text.lower())
            word_counts.update(words)

        top_n = 100
        top_words = word_counts.most_common(top_n)

        gender_counts = Counter()
        race_counts = Counter()
        age_counts = Counter()

        for d in data:
            gender_counts.update([d['gender']])
            race_counts.update([d['race']])
            age_counts.update([d['age']])

        gender_percentages = {gender: (count / len(data)) * 100 for gender, count in gender_counts.items()}
        race_percentages = {race: (count / len(data)) * 100 for race, count in race_counts.items()}
        age_percentages = {age: (count / len(data)) * 100 for age, count in age_counts.items()}

        with open("dataset_analysis.txt", 'w') as file:
            file.write("Top {} most common words:\n".format(top_n))
            for word, count in top_words:
                file.write("{}: {}\n".format(word, count))

            file.write("Gender Counts: {}\n".format(gender_percentages))
            file.write("Race Counts: {}\n".format(race_percentages))
            file.write("Age Counts: {}\n".format(age_percentages))

if __name__ == "__main__":
    dataset_analysis = DatasetAnalysis()
    dataset_analysis()