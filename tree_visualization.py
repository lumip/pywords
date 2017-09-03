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

from typing import List

import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

from input_parsing import WordProcessor
from training_data_structures import Cluster

__all__ = ["visualize_tree"]


def make_leaf_label(class_index: int, clusters: List[Cluster], input_processor: WordProcessor) -> str:
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


def make_node_label(feature: int, vectorizer: DictVectorizer) -> str:
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


def visualize_tree(classifier: DecisionTreeClassifier,
                   input_processor: WordProcessor,
                   vectorizer: DictVectorizer,
                   clusters: List[Cluster],
                   file_name: str,
                   format: str = "svg") -> None:
    tree = classifier.tree_
    graph = graphviz.Digraph(format=format)
    graph.node_attr.update(shape="box", style="rounded, filled", color="black", fontname="helvetica")
    graph.edge_attr.update(fontname="helvetica")
    for i in range(tree.node_count):
        is_leaf = (tree.children_left[i] < 0)
        if is_leaf:
            value = tree.value[i]
            c = next(c for c, n in enumerate(value[0]) if n > 0)
            graph.node(str(i), label="{}".format(make_leaf_label(c, clusters, input_processor)), fillcolor="#" + background_colors[c] + "AA",
                       margin="0.2")
        else:
            graph.node(str(i), label="{}".format(make_node_label(tree.feature[i], vectorizer)), fillcolor="#FFFFFFFF")
            graph.edge(str(i), str(tree.children_left[i]), label="False", labeldistance="2.5")
            graph.edge(str(i), str(tree.children_right[i]), label="True", labeldistance="2.5")
    graph.render(file_name)
