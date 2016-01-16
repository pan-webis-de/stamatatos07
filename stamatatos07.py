from collections import Counter

import argparse
import jsonhandler
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


def create_ranking(n, L, method="d1"):
# If you want to do training:
    bigram_profile = []
    counts = []     # summ of all n-gram
    if method == "d2":
        norm_text = ''
    for cand in jsonhandler.candidates:
        text = ''
        for file in jsonhandler.trainings[cand]:
                    # Get content of training file 'file' of candidate 'cand'
                    # as a string with:
                text = text + jsonhandler.getTrainingText(cand, file)
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

    for file in jsonhandler.unknowns:
        result = []
        # Get content of unknown file 'file' as a string with:
        test = ''
        test = jsonhandler.getUnknownText(file)
        # Determine author of the file, and score (optional)
        bigram_all = Counter(find_ngrams(test, n))
        counts_test = sum(bigram_all.values())
        bigram_test = Counter(dict(bigram_all.most_common(L)))

        for cand_nu in range(len(jsonhandler.candidates)):
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
        author = jsonhandler.candidates[result.index(min(result))]

#    author = "oneAuthor"
        score = 1
        logging.debug("%s attributed to %s", file, author)
        authors.append(author)
        scores.append(score)
    return (authors, scores)


def fit_parameters():
    n_range = [3, 4, 5, 6]
    L_range = [500, 1000, 2000, 3000, 5000]
#    n_range = [2,3]
#    L_range = [20, 50, 100]
    jsonhandler.loadGroundTruth()
    results = []
    for n in n_range:
        for L in L_range:
            logging.info("Test parameters: n=%d, l=%d", n, L)
            authors, scores = create_ranking(n, L)
            evaluation = evalTesting(jsonhandler.unknowns, authors)
            results.append((evaluation["accuracy"], n, L))
    return results

def evalTesting(texts, cands, scores=None):
    succ = 0
    fail = 0
    sucscore = 0
    failscore = 0
    for i in range(len(texts)):
        if jsonhandler.trueAuthors[i] == cands[i]:
            succ += 1
            if scores != None:
                sucscore += scores[i]
        else:
            fail += 1
            if scores != None:
                failscore += scores[i]
    result = {"fail": fail, "success": succ, "accuracy":
              succ / float(succ + fail)}
    return result

def optimize(corpusdir, outputdir):
    parameters = fit_parameters()
    acc, n, L = max(parameters, key=lambda r: r[0])
    logging.info("Choose parameters: n=%d, l=%d", n, L)
    logging.disable(logging.DEBUG)
    authors, scores = create_ranking(n, L)
    jsonhandler.storeJson(outputdir, jsonhandler.unknowns, authors, scores)


def test_method(corpusdir, outputdir, method="d1", n=3, L=2000):
    logging.info("Test method %s with L=%d", method, L)
    authors, scores = create_ranking(n, L, method)
    jsonhandler.storeJson(outputdir, jsonhandler.unknowns, authors, scores)


def compare_methods(corpusdir, outputdir):
    n = 3
    logging.disable(logging.DEBUG)
    for L in range(500, 10500, 500):
        for m in ["d0", "d1", "d2", "SPI"]:
            test_method(corpusdir, outputdir, method=m, n=n, L=L)


def main():
    parser = argparse.ArgumentParser(description='Tira submission for' +
                                     ' "Author identification using imbalanced and limited training texts."')
    parser.add_argument('-i',
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o',
                        action='store',
                        help='Path to output directory')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']

    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()
    
    test_method(corpusdir, outputdir)

if __name__ == "__main__":
    # execute only if run as a script
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()
