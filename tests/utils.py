import json
from numpy import long, ndarray
from gem_metrics.texts import Texts

def assertDeepAlmostEqual(test_case, expected, actual, *args, **kwargs):
    """
    Taken from https://github.com/larsbutler/oq-engine/blob/master/tests/utils/helpers.py
    Assert that two complex structures have almost equal contents.
    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).
    :param test_case: TestCase object on which we can call all of the basic
        'assert' methods.
    :type test_case: :py:class:`unittest.TestCase` object
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, long, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(test_case, v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                assertDeepAlmostEqual(test_case, expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            test_case.assertEqual(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError(f"%{exc.args}\nTRACE: {trace}")
        raise exc

def read_test_data(pth: str, data_type: Texts) -> Texts:
    """Given path to a test dataset file, returns an object of type Texts.

    Args:
        pth (str): [Path to the dataset]
        data_type (Texts): Predictions, References, or Sources

    Returns:
        Texts: An object of type either Predictions, References, or Sources 
    """
    with open(pth, "r") as fin:
        return data_type(json.load(fin))