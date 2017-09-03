import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

import input_parsing as par
from training_data_structures import TrainingSetElement, ClusterSet
from tree_visualization import visualize_tree

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
visualize_tree(classifier, input_processor, vectorizer, clusters, "rules")



print("done!")
