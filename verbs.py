import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import graphviz

import input_parsing as par
from training_data_structures import TrainingSetElement, ClusterSet

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

print("Clustering training word pairs by local transformations...")
clusters = ClusterSet()
for training_instance in training_set:
     clusters.add(training_instance)

clusters = clusters.get_clusters()

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

vectorizer = DictVectorizer()
x_data = vectorizer.fit_transform(x_data)

print("... extracted {} features for training the classifier".format(len(vectorizer.get_feature_names())))

print("Training classifier....")
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_data, c_data)

print("Creating tree visualization...")

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
