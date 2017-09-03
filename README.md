# pywords

An implementation of decision tree learning to derive rules for words transformations in natural languages (e.g. verb conjugations).

Currently under early development.

Implemented features:

- batch learning from a list of word pairs (not very sophisticated; no cross validation and training/test set split yet)

### Usage

Prepare the training data as word pairs of base form and transformed form in a simple text file, one word pair per line separated by a comma (,). Refer to words.txt, words2.txt and words_kor.txt for examples.

Invoke the learning algorithm with:

pywords-train.py <input_filename>

it will read all words from the given file, build the decision tree and store it as "classifier.clf"

Optional parameters to be inserted before input_filename:

- -v or --visualize : creates an SVG file showing the generated decision tree in a human readable fashion
- -o <output_filename> or --outfile=<output_filename> : sets the name of the output file. Default is "classifier". The file ending ".clf" is added in any case.
- --no_saveout: do not store the trained classifier to disk (does not affect the visualization if -v or --visualize is also given)

Current dependencies for running:

- pygraphviz
- sklearn