# collection of inputs for testing various metrics
from gem_metrics.texts import Predictions, References, Sources
from tests.utils import read_test_data


class TestData:

    sources = read_test_data("test_data/unit_tests/sources.json", Sources)
    references = read_test_data(
        "test_data/unit_tests/references.json", References)

    empty_references = read_test_data(
        "test_data/unit_tests/empty_references.json", References)

    predictions = read_test_data(
        "test_data/unit_tests/predictions.json", Predictions)

    # predictions same as the references
    identical_predictions = read_test_data(
        "test_data/unit_tests/identical_predictions.json", Predictions)

    # empty predictions
    empty_predictions = read_test_data(
        "test_data/unit_tests/empty_predictions.json", Predictions)

    # predictions set to references reversed and punctuation removed.
    reversed_predictions = read_test_data(
        "test_data/unit_tests/reversed_predictions.json", Predictions)
