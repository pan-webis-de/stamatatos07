# stamatatos07
Stamatatos, E. (2007). Author identification using imbalanced and limited training texts. In Proceedings of the 4th International Workshop on Text-based Information Retrieval (pp. 237- 241).

Dissimilarity function, where only n-grams from the test text profile are considered. 

# How to use this implementation
This code is written in Python 2.7

To run the main program of n_gram.py, both files (jsonhandler.py and n_gram.py) must be stored in the same folder together with a subfolder which is called in the last line of n_gram.py.

The subfolder must include:
- candidates' trainings set
- set of unknown texts
- meta-file

The results are saved to the json-file out.json in in the subfolder.
