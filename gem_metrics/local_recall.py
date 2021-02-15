from .metric import ReferencedMetric
from collections import Counter, defaultdict


class LocalRecall(ReferencedMetric):
    """
    LocalRecall checks the extent to which a model produces the same tokens as the reference data.

    For each item, tokens receive an importance score. If all N annotators use a particular word,
    that word gets an importance score of N.

    The output of this metric is a dictionary with {1:score, ..., N: score}.

    The local recall metric is based on Van Miltenburg et al. (2018).
    Paper: https://www.aclweb.org/anthology/C18-1147/
    Repository: https://github.com/evanmiltenburg/MeasureDiversity/blob/master/local_recall.py

    The main difference is that Van Miltenburg et al. only include content words,
    while the code below just counts ALL tokens, including determiners (a, the) etc.

    This means that the scores produced by this code will be higher than the ones produced by the original code.
    The advantage is that we don't have to rely on a part-of-speech tagger.
    """

    def compute(self, predictions, references):
        results = LocalRecall.local_recall_scores(predictions.list_tokenized_lower_nopunct,
                                                  references.list_tokenized_lower_nopunct)

        return {'local_recall': results}

    @staticmethod
    def build_reference_index(refs):
        """
        Build reference index for a given item.
        Input: list of lists (list of sentences, where each sentence is a list of string tokens).
        Output: dictionary with key: int (1-number of references), value: set of words.
        """
        counts = Counter()
        for ref in refs:
            counts.update(set(ref))
        importance_index = defaultdict(set)
        for word, count in counts.items():
            importance_index[count].add(word)
        return importance_index

    @staticmethod
    def check_item(prediction, refs):
        """
        Check whether the predictions capture words that are frequently mentioned.

        This function produces more info than strictly needed.
        Use the detailed results to analyze system performance.
        """
        reference_index = LocalRecall.build_reference_index(refs)
        pred_tokens = set(prediction)
        results = dict()
        for n in range(1, len(refs) + 1):
            overlap = pred_tokens & reference_index[n]
            results[f'overlap-{n}'] = overlap
            results[f'size-overlap-{n}'] = len(overlap)
            results[f'refs-{n}'] = reference_index[n]
            results[f'size-refs-{n}'] = len(reference_index[n])
            # Just in case there are no words at all that occur in all references,
            # Make score equal to None to avoid divide by zero error.
            # This also avoids ambiguity between "no items recalled" and "no items to recall".
            if len(reference_index[n]) > 0:
                results[f'item-score-{n}'] = len(overlap) / len(reference_index[n])
            else:
                results[f'item-score-{n}'] = None
        return results

    @staticmethod
    def replace(a_list, to_replace, replacement):
        """
        Returns a_list with all occurrences of to_replace replaced with replacement.
        """
        return [replacement if x == to_replace else x for x in a_list]

    @staticmethod
    def aggregate_score(outcomes):
        """
        Produce an aggregate score based on a list of tuples: [(size_overlap, size_refs)]
        """
        overlaps, ref_numbers = zip(*outcomes)
        ref_numbers = LocalRecall.replace(ref_numbers, None, 0)
        score = (sum(overlaps) / sum(ref_numbers)) if sum(ref_numbers) > 0 else 0
        return score

    @staticmethod
    def local_recall_scores(predictions, full_references):
        """
        Compute local recall scores.
        """
        num_refs = set()
        outcomes = defaultdict(list)
        for pred, refs in zip(predictions, full_references):
            results = LocalRecall.check_item(pred, refs)
            total_refs = len(refs)
            num_refs.add(total_refs)
            for n in range(1, total_refs + 1):
                pair = (results[f'size-overlap-{n}'], results[f'size-refs-{n}'])
                outcomes[n].append(pair)
        scores = {n: LocalRecall.aggregate_score(outcomes[n]) for n in range(1, max(num_refs) + 1)}
        return scores
