# pywords - A machine learning implementation for words transformations in natural languages (e.g. verb conjugations) using decision trees
# Copyright (C) 2017  Lukas Prediger <lukas.prediger@rwth-aachen.>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>


import codecs
import sys
import getopt
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

import input_parsing as par
from training_data_structures import TrainingSetElement, ClusterSet
from tree_visualization import visualize_tree


def exit_with_usage():
    print("usage: {} [-v|--visualize] [-o <output_file>|--outfile=<output_file>] [no_saveout] <input_file>".format(sys.argv[0]))
    sys.exit(2)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvo:", ["outfile=", "visualize", "no_saveout"])
    except getopt.GetoptError:
        exit_with_usage()

    save_classifier = True
    create_visualization = False
    output_name = "classifier"
    for opt, arg in opts:
        if opt == "--no_saveout":
            save_classifier = False
        elif opt == "--visualize" or opt == "-v":
            create_visualization = True
        elif opt == "--outfile" or opt == "-o":
            output_name = arg
        elif opt == "-h":
            exit_with_usage()

    if len(args) == 0:
        exit_with_usage()

    input_name = args[0]

    print("Loading training word pairs...")
    input_processor = par.CombinedProcessor([par.StripProcessor(), par.HangeulComposer()])
    word_pairs = []
    with codecs.open(input_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_pairs.append(tuple(input_processor.process_input(part) for part in line.split(",")))
    print("... read {} word pairs".format(len(word_pairs)))

    print("Analyzing training word pairs...")
    training_set = [TrainingSetElement(pair[0], pair[1]) for pair in word_pairs]

    print("Clustering training word pairs by local transformations...")
    clusters = ClusterSet()
    for training_instance in training_set:
        clusters.add(training_instance)
    clusters = clusters.get_clusters()
    print("... split word pairs into {} clusters of similar transformations".format(len(clusters)))

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

    if save_classifier:
        # sklearn advises to use pickle to store classifiers: http://scikit-learn.org/stable/modules/model_persistence.html
        print("Storing classifier...")
        with open(output_name + ".clf", "wb") as output_file:
            pickle.dump(classifier, output_file)

    if create_visualization:
        print("Creating tree visualization...")
        visualize_tree(classifier, input_processor, vectorizer, clusters, output_name)

    print("done!")

if __name__ == "__main__":
    main(sys.argv[1:])