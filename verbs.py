import sys
import codecs
from typing import TypeVar, Tuple, List
#import tensorflow as tf
#from sklearn.tree import DecisionTreeClassifier

import word_analysis as ana
from cluster_set import ClusterSet

class TrainingSetElement:

    def __init__(self, word_a: str, word_b: str) -> None:
        self.__word_a = word_a
        self.__word_b = word_b
        self.__subsequence_intervals = ana.analyze_word_pair(self.__word_a, self.__word_b)

    @property
    def word_a(self) -> str:
        return self.__word_a

    @property
    def word_b(self) -> str:
        return self.__word_b

    @property
    def subsequence_intervals(self) -> ana.WordSubsequenceIntervals:
        return self.__subsequence_intervals

    def __hash__(self) -> int:
        return hash(self.__subsequence_intervals)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TrainingSetElement): return False
        return other.subsequence_intervals == self.subsequence_intervals

    def __repr__(self) -> str:
        return "(" + self.word_a + ", " + self.word_b + ", " + str(self.subsequence_intervals) + ")"

print("Loading training word pairs...")
word_pairs = set()
with codecs.open("words.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word_pairs.add(tuple(part.strip() for part in line.split(",")))


print("Analyzing training word pairs...")
training_set = [TrainingSetElement(pair[0], pair[1]) for pair in word_pairs]

print(str(training_set))

print("Clustering training word pairs...")
clusters = ClusterSet()
for elem in training_set:
    clusters.add(elem)

print(clusters.get_clusters())

# todo: make transformations invariant to skip lengths (same replacements and inserts -> covered by same transformation for different skip lengths)?
# todo: use ML to discover similar rules for the above?
