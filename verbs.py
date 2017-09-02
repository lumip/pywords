import codecs
from typing import Set
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

import word_analysis as ana
import input_parsing as par

# todo: split Korean (or other) compound runes into single character runes by detecting and preprocessing
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
        return "({}, {}, {})".format(self.word_a, self.word_b, repr(self.transformation))

input_processor = par.CombinedProcessor([par.HangeulComposer()])

print("Loading training word pairs...")
word_pairs = []
with codecs.open("words_kor.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word_pairs.append(tuple(input_processor.process_input(part.strip()) for part in line.split(",")))

print("... read {} word pairs".format(len(word_pairs)))

print("Analyzing training word pairs...")
training_set = [TrainingSetElement(pair[0], pair[1]) for pair in word_pairs]

print(training_set)

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
print(clusters)

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

print("Creating tree visualization...")

import graphviz
#class_names = list(input_processor.process_output(str(cluster.transformation)) for cluster in clusters)
#class_names = list("{} : {} -> {}".format(str(rule), training_instance.word_a, training_instance.word_b) for rule, training_instance in ((cluster.transformation, cluster.items.pop()) for cluster in clusters))
#dot_data = export_graphviz(classifier,
#                           out_file=None,
#                           feature_names=vectorizer.get_feature_names(),
#                           class_names=class_names,
#                           filled=True,
#                           rounded=True)
#graph = graphviz.Source(dot_data)
#graph.format = "svg"
#graph.render("classifier")

def make_leaf_label(class_index) -> str:
    cluster = clusters[class_index]
    transf = cluster.transformation
    training_instance = cluster.items.pop()
    return "{}\n e.g. {} -> {}\n(applies to {} instances)".format(
        input_processor.process_output(str(transf).replace(",",",\n")),
        input_processor.process_output(training_instance.word_a),
        input_processor.process_output(training_instance.word_b),
        len(cluster.items)
    )

def count_suffix(nr) -> str:
    if nr % 10 == 1 and nr != 11:
        return "1st"
    elif nr % 10 == 2 and nr != 12:
        return "2nd"
    elif nr % 10 == 3 and nr != 13:
        return "3rd"
    else:
        return "{}th".format(nr)

def make_node_label(feature) -> str:
    feature_name = vectorizer.get_feature_names()[feature]
    a = feature_name.split(vectorizer.separator)
    if a[0] == "length":
        return "<word length dependence>"
    ind = int(a[0])
    char = a[1]
    return "is {} letter from {} a {}?".format(count_suffix(ind+1) if ind >= 0 else count_suffix(-ind), "front" if ind >= 0 else "back", char)

background_colors = [
    "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF",
    "800000", "008000", "000080", "808000", "800080", "008080", "808080",
    "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
    "400000", "004000", "000040", "404000", "400040", "004040", "404040",
    "200000", "002000", "000020", "202000", "200020", "002020", "202020",
    "600000", "006000", "000060", "606000", "600060", "006060", "606060",
    "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
    "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0",
]

graph = graphviz.Digraph(format="svg")
graph.node_attr.update(shape="box", style="rounded, filled", color="black", fontname="helvetica")
graph.edge_attr.update(fontname="helvetica")
t = classifier.tree_ # type: BaseDecisionTree
for i in range(t.node_count):
    is_leaf = (t.children_left[i] < 0)
    if is_leaf:
        value = t.value[i]
        c = next(c for c, n in enumerate(value[0]) if n > 0)
        graph.node(str(i), label="{}".format(make_leaf_label(c)), fillcolor="#" + background_colors[c] + "AA", margin="0.2")
    else:
        graph.node(str(i), label="{}".format(make_node_label(t.feature[i])), fillcolor="#FFFFFFFF")
        graph.edge(str(i), str(t.children_left[i]), label="False", labeldistance="2.5")
        graph.edge(str(i), str(t.children_right[i]), label="True", labeldistance="2.5")
graph.render("rules")

print("done!")


# print("Clustering training word pairs by local transformations...")
# clusters = ClusterSet()
# for elem in training_set:
#     clusters.add(elem)
#
# clusters = clusters.get_clusters()
# print(clusters)
