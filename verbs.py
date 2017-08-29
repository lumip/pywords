import sys
import codecs
from typing import TypeVar, Tuple, List, Set
#import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

import word_analysis as ana
from cluster_set import ClusterSet

class TrainingSetElement:

    def __init__(self, word_a: str, word_b: str) -> None:
        self.__word_a = word_a
        self.__word_b = word_b
        lcs_matrix = ana.LCSMatrix(word_a, word_b)
        subsequence_intervals = ana.WordSubsequenceIntervals(lcs_matrix)
        transformation = ana.build_word_transformation(subsequence_intervals)
        self.__subsequence_intervals = subsequence_intervals
        self.__transformation = transformation

    @property
    def word_a(self) -> str:
        return self.__word_a

    @property
    def word_b(self) -> str:
        return self.__word_b

    @property
    def transformation(self) -> ana.WordTransformation:
        return self.__transformation

    @property
    def subsequence_intervals(self) -> ana.WordSubsequenceIntervals:
        return self.__subsequence_intervals

    # def __hash__(self) -> int:
    #     return hash(self.__subsequence_intervals)
    #
    # def __eq__(self, other) -> bool:
    #     if not isinstance(other, TrainingSetElement): return False
    #     return other.subsequence_intervals == self.subsequence_intervals

    def __repr__(self) -> str:
        return "(" + self.word_a + ", " + self.word_b + ", " + str(self.transformation) + ")"

print("Loading training word pairs...")
#word_pairs = set()
word_pairs = []
with codecs.open("words2.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        #word_pairs.add(tuple(part.strip() for part in line.split(",")))
        word_pairs.append(tuple(part.strip() for part in line.split(",")))

print("... read {} word pairs".format(len(word_pairs)))

print("Analyzing training word pairs...")
training_set = [TrainingSetElement(pair[0], pair[1]) for pair in word_pairs]

#print(str(training_set))

class Cluster:

    def __init__(self, first_item: TrainingSetElement):
        self.__transformation = first_item.transformation # type: ana.WordTransformation
        self.__items = {first_item} # type: Set[TrainingSetElement]

    def can_add_item(self, item: TrainingSetElement) -> bool:
        if self.__transformation.maybe_joinable(item.transformation):
            joined_transformation = self.__transformation.join(item.transformation)
            for e in self.__items:
                if joined_transformation.apply(e.word_a) != e.word_b:
                    return False
            if joined_transformation.apply(item.word_a) != item.word_b:
                return False
            return True
        return False

    def add_item(self, item: TrainingSetElement) -> bool:
        if self.can_add_item(item):
            joined_transformation = self.__transformation.join(item.transformation)
            self.__transformation = joined_transformation
            self.__items.add(item)
            return True
        return False

    @property
    def transformation(self) -> ana.WordTransformation:
        return self.__transformation

    @property
    def items(self):
        return self.__items.copy()

    def __repr__(self) -> str:
        return "<Cluster, {}, [{}] {} elements>".format(str(self.transformation), self.__items, len(self.__items))

print("Clustering training word pairs by local transformations...")
clusters = []
for training_instance in training_set:
    found_cluster = False
    for cluster in clusters:
        if cluster.add_item(training_instance):
            found_cluster = True
            break
    if not found_cluster:
        clusters.append(Cluster(training_instance))

print("... split word pairs into {} clusters of similar transformations".format(len(clusters)))
#print(clusters)

# todo: [done] make transformations invariant to skip lengths (same replacements and inserts -> covered by same transformation for different skip lengths)?
# todo: use ML to discover similar rules for the above instead of hardcoding?

print("Extracting features for training...")

x_data = []
c_data = []
for c, cluster in enumerate(clusters):
    for training_instance in cluster.items:
        features = dict()
        word = training_instance.word_a
        length = len(word)
        features["length"] = length
        for i in range(length):
            features[i] = word[i]
            features[i - length] = word[i]
        x_data.append(features)
        c_data.append(c)

#print(x_data)
vectorizer = DictVectorizer()
x_data = vectorizer.fit_transform(x_data)

print("... extracted {} features for training the classifier".format(len(vectorizer.get_feature_names())))

print("Training classifier....")
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_data, c_data)

import graphviz
class_names = list(str(cluster.transformation) for cluster in clusters)
dot_data = export_graphviz(classifier,
                           out_file=None,
                           feature_names=vectorizer.get_feature_names(),
                           class_names=class_names,
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("classifier")


# print("Clustering training word pairs by local transformations...")
# clusters = ClusterSet()
# for elem in training_set:
#     clusters.add(elem)
#
# clusters = clusters.get_clusters()
# print(clusters)
