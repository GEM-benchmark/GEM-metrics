import argparse
import json
from pprint import pprint
from datetime import datetime
from string import punctuation
from nltk import word_tokenize, ngrams
import random

random.seed(1234)
PUNCTUATION = set(punctuation)


def get_vocabulary(tokenized_data):
    "Compute vocabulary, based on the tokenized data."
    vocab = set()
    for sent in tokenized_data:
        vocab.update(sent)
    return vocab


def get_ngram_vocabulary(tokenized_data, n):
    "Compute n-gram vocabulary, based on the tokenized data."
    vocab = set()
    for sent in tokenized_data:
        vocab.update(ngrams(sent,n))
    return vocab


def tokenize(sentence, language, punctuation):
    "Tokenize sentence, optionally excluding punctuation."
    tokens = word_tokenize(sentence, language)
    if not punctuation:
        tokens = [token for token in tokens if not token in PUNCTUATION]
    return tokens


def load_data(filename, language, punctuation, case_sensitive):
    "Load the data as a list of lists of strings."
    if case_sensitive:
        with open(filename) as f:
            main_data = [tokenize(line, language, punctuation) for line in f]
    else:
        with open(filename) as f:
            main_data = [tokenize(line.lower(), language, punctuation) for line in f]
    return main_data


def TTR(list_of_words):
    "Compute type-token ratio."
    return len(set(list_of_words))/len(list_of_words)


def MSTTR(tokenized_data, window_size):
    "Compute Mean-Segmental Type-Token Ratio (MSTTR; Johnson, 1944)."
    chunk = []
    ttrs = []
    for sentence in tokenized_data:
        chunk_length = len(chunk)
        sentence_length = len(sentence)
        combined = chunk_length + sentence_length
        if combined < window_size:
            chunk.extend(sentence)
        elif combined == window_size:
            chunk.extend(sentence)
            ttrs.append(TTR(chunk))
            chunk = []
        else:
            needed = window_size - chunk_length
            chunk.extend(sentence[:needed])
            ttrs.append(TTR(chunk))
            chunk = sentence[needed:]
    results = dict(msttr_value = sum(ttrs)/len(ttrs),
                   num_ttrs = len(ttrs),
                   ttrs = ttrs)
    return results


def repeated_MSTTR(tokenized_data, window_size, repeats=5):
    "Repeated MSTTR to obtain a more robust MSTTR value."
    msttrs = []
    for i in range(repeats):
        sentences = random.sample(tokenized_data, len(tokenized_data))
        msttr_results = MSTTR(sentences, window_size)
        msttrs.append(msttr_results['msttr_value'])
    results = sum(msttrs)/len(msttrs)
    return results


def num_tokens(tokenized_data):
    "Compute the number of tokens."
    return sum(len(sentence) for sentence in tokenized_data)


def num_ngram_tokens(tokenized_data, n):
    "Compute the number of tokens."
    return sum(len(list(ngrams(sentence,n))) for sentence in tokenized_data)


def main(filename, 
         language='english', 
         punctuation=False, 
         output_vocab=False, 
         case_sensitive=False,
         msttr_window=100,
         repeat_MSTTR=False,
         num_repeats=5):
    """
    Main function that computes all statistics and returns a dictionary containing
    the results for the given input file.
    """
    results = dict()
    main_data = load_data(filename, language, punctuation, case_sensitive)
    vocabulary = get_vocabulary(main_data)
    
    results['vocab_size'] = len(vocabulary)
    results['num_bigram_types'] = len(get_ngram_vocabulary(main_data, 2))
    results['num_bigram_tokens'] = num_ngram_tokens(main_data, 2)
    results['MSTTR'] = MSTTR(main_data, msttr_window)
    
    if repeat_MSTTR:
        results['Repeated_MSTTR'] = repeated_MSTTR(main_data, msttr_window, repeats=5)
    
    if output_vocab:
        results['vocab'] = list(vocabulary)
    
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute diversity statistics.')
    parser.add_argument('--language', 
                        default='english', 
                        type=str, 
                        help="Any language that has a model in nltk_data/tokenizers/punkt.")
    parser.add_argument('--msttr_window', 
                        default=100, 
                        type=int, 
                        help="Window size to compute the mean-segmental type-token ratio.")
    parser.add_argument('--punctuation',
                        action='store_true',
                        help="Include punctuation in the statistics.")
    parser.add_argument('--output_vocab',
                        action='store_true',
                        help="Include vocabulary in the results file.")
    parser.add_argument('--case_sensitive',
                        action='store_true',
                        help="Make the analysis case sensitive?")
    parser.add_argument('--repeat_MSTTR',
                        action='store_true',
                        help="Carry out repeated MSTTR?")
    parser.add_argument('--num_repeats',
                        default=5,
                        type=int,
                        help="Number of times to carry out repeated MSTTR.")
    parser.add_argument('filename', help="File you want to analyse")
    args = parser.parse_args()
    settings = vars(args)
    
    # Visual confirmation:
    print("Running with the following settings:")
    pprint(settings)
    
    # Compute results:
    results = main(**settings)
    
    # Add results to general configuration information:
    settings.update(results)
    
    # Create specific filename.
    filename = datetime.now().strftime("Diversity_results_%Y-%m-%d_at_%H-%M-%S.json")
    print("Saving data to:", filename)
    
    # Save data.
    with open(filename, 'w') as f:
        json.dump(settings, f, indent=4)
