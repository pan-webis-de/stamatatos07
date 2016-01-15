from jsonhandler import Jsonhandler
from collections import Counter

import argparse
import logging


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def d0(corpus_profile, corpus_size, unknown_profile, unknown_size):
    keys = set(unknown_profile.keys()) | set(corpus_profile.keys())
    summe = 0.0
    for k in keys:
        f1 = float(corpus_profile[k]) / corpus_size
        f2 = float(unknown_profile[k]) / unknown_size
        summe = summe + (2 * (f1 - f2) / (f1 + f2)) ** 2
    return summe


def d1(corpus_profile, corpus_size, unknown_profile, unknown_size):
    keys = set(unknown_profile.keys())
    summe = 0.0
    for k in keys:
        f1 = float(corpus_profile[k]) / corpus_size
        f2 = float(unknown_profile[k]) / unknown_size
        summe = summe + (2 * (f1 - f2) / (f1 + f2)) ** 2
    return summe


def d2(corpus_profile, corpus_size, unknown_profile, unknown_size,
        norm_profile, norm_size):
    keys = set(unknown_profile.keys())
    summe = 0.0
    for k in keys:
        f1 = float(corpus_profile[k]) / corpus_size
        f2 = float(unknown_profile[k]) / unknown_size
        f3 = float(norm_profile[k]) / norm_size

        summe = summe + \
            (2 * (f1 - f2) / (f1 + f2)) ** 2 * (2 * (f2 - f3) / (f2 + f3)) ** 2
    return summe


def SPI(corpus_profile, unknown_profile):
    return -len(set(unknown_profile.keys()) &
                set(corpus_profile.keys()))


def create_ranking(handler, n, L, method="d1"):
# If you want to do training:
    bigram_profile = []
    counts = []     # summ of all n-gram
    if method == "d2":
        norm_text = ''
    for cand in handler.candidates:
        text = ''
        for file in handler.trainings[cand]:
                    # Get content of training file 'file' of candidate 'cand'
                    # as a string with:
                text = text + handler.getTrainingText(cand, file)
        bigram_all = Counter(find_ngrams(text, n))

        counts.append(sum(bigram_all.values()))
        bigram_profile.append(Counter(dict(bigram_all.most_common(L))))
        if method == "d2":
            norm_text = norm_text + text
        text = ''
    if method == "d2":
        norm_all = Counter(find_ngrams(norm_text, n))
        norm_size = sum(norm_all.values())
        norm_profile = Counter(dict(norm_all.most_common(L)))

# Create lists for your answers (and scores)
    authors = []
    scores = []

    for file in handler.unknowns:
        result = []
        # Get content of unknown file 'file' as a string with:
        test = ''
        test = handler.getUnknownText(file)
        # Determine author of the file, and score (optional)
        bigram_all = Counter(find_ngrams(test, n))
        counts_test = sum(bigram_all.values())
        bigram_test = Counter(dict(bigram_all.most_common(L)))

        for cand_nu in range(len(handler.candidates)):
            dissimilarity = 0
            if method == "d0":
                dissimilarity = d0(bigram_profile[cand_nu],
                                   counts[cand_nu], bigram_test, counts_test)
            elif method == "d1":
                dissimilarity = d1(bigram_profile[cand_nu],
                                   counts[cand_nu], bigram_test, counts_test)
            elif method == "d2":
                dissimilarity = d2(bigram_profile[cand_nu],
                                   counts[cand_nu], bigram_test, counts_test,
                                   norm_profile, norm_size)
            elif method == "SPI":
                dissimilarity = SPI(bigram_profile[cand_nu], bigram_test)
            else:
                raise Exception("unknown method for create_ranking")
            result.append(dissimilarity)
        author = handler.candidates[result.index(min(result))]

#    author = "oneAuthor"
        score = 1
        logging.debug("%s attributed to %s", file, author)
        authors.append(author)
        scores.append(score)
    return (authors, scores)


def fit_parameters(handler):
    n_range = [3, 4, 5, 6]
    L_range = [500, 1000, 2000, 3000, 5000]
#    n_range = [2,3]
#    L_range = [20, 50, 100]
    handler.loadTraining()
    results = []
    for n in n_range:
        for L in L_range:
            logging.info("Test parameters: n=%d, l=%d", n, L)
            authors, scores = create_ranking(handler, n, L)
            evaluation = handler.evalTesting(handler.unknowns, authors)
            results.append((evaluation["accuracy"], n, L))
    return results


def tira(corpusdir, outputdir):
    handler = Jsonhandler(corpusdir, out="stamatatos07.json")
    parameters = fit_parameters(handler)
    acc, n, L = max(parameters, key=lambda r: r[0])
    logging.info("Choose parameters: n=%d, l=%d", n, L)
    logging.disable(logging.DEBUG)
    handler.loadTesting()
    authors, scores = create_ranking(handler, n, L)
    handler.storeJson(handler.unknowns, authors, scores, path=
                      outputdir)


def test_method(corpusdir, outputdir, method="d1", n=3, L=2000):
    logging.info("Test method %s with L=%d", method, L)
    handler = Jsonhandler(corpusdir, out=method + "L" + str(L) + ".json")
    handler.loadTesting()
    authors, scores = create_ranking(handler, n, L, method)
    handler.storeJson(handler.unknowns, authors, scores, path=
                      outputdir)


def compare_methods(corpusdir, outputdir):
    handler = Jsonhandler(corpusdir, out="stamatatos07.json")
    n = 3
    logging.disable(logging.DEBUG)
    for L in range(500, 10500, 500):
        for m in ["d0", "d1", "d2", "SPI"]:
            test_method(corpusdir, outputdir, method=m, n=n, L=L)


def main():
    parser = argparse.ArgumentParser(description='Tira submission for' +
                                     ' "Author identification using imbalanced and limited training texts."')
    parser.add_argument('length', type=int,
                        help="size of profiles")
    parser.add_argument('-i',
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o',
                        action='store',
                        help='Path to output directory')

    parser.add_argument('-m',
                        action='store',
                        help='Method (one of d0,d1,d2,SPI)')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']
    method = args['m']

    # tira(corpusdir, outputdir)
    test_method(corpusdir, outputdir, method, L=args['length'])

if __name__ == "__main__":
    # execute only if run as a script
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()
