import sys
import codecs
#import tensorflow as tf

import word_analysis as ana

print("Loading training word pairs...")
word_pairs = []
with codecs.open("words.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word_pairs.append([part.strip() for part in line.split(",")])


print("Analyzing training word pairs...")
training_set = [(pair[0], pair[1], ana.analyze_word_pair(pair[0], pair[1])) for pair in word_pairs]

print(str(training_set))

print("Clustering training word pairs...")
clusters = dict()
for elem in training_set:
    id = hash(elem[2])
    if id not in clusters:
        clusters[id] = [elem]
    else:
        clusters[id].append(elem) # todo: needs equivalence checks and separate lists for non-equivalent elems with same hash

print(clusters)

# todo: make transformations invariant to skip lengths (same replacements and inserts -> covered by same transformation for different skip lengths)?
# todo: use ML to discover similar rules for the above?
