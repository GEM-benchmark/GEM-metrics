# collection of inputs for testing various metrics
from gem_metrics.texts import Predictions, References, Sources


class TestData:

    sources =  Sources({"values": ["Alimentum is a non family-friendly restaurant near Burger King in the city centre.",
                                          "Alimentum is located in the city centre. It is not family-friendly.",
                                          "Or Orleans has a home."
                                          ], "language": "en"})
    references = References({"values": [
        {
            "target": ["Alimentum is not family-friendly, and is near the Burger King in the city centre."]
        },
        {
            "target": ["There is a place in the city centre, Alimentum, that is not family-friendly."]
        },
        {
            "target": ["There is a house in New Orleans."]
        }
    ], "language": "en"})

    empty_references = References({"values": [
        "",
        "",
        ""
    ], "language": "en"})

    predictions = Predictions({"values": ["Alimentum is a non family-friendly restaurant near Burger King in the city centre.",
                                          "Alimentum is located in the city centre. It is not family-friendly.",
                                          "Or Orleans has a home."
                                          ], "language": "en"})

    # predictions same as the references
    identical_predictions = Predictions({"values": [
        "Alimentum is not family-friendly, and is near the Burger King in the city centre.",
        "There is a place in the city centre, Alimentum, that is not family-friendly.",
        "There is a house in New Orleans."
    ], "language": "en"})

    # empty predictions
    empty_predictions = Predictions({"values": [
        "",
        "",
        ""
    ], "language": "en"})

    # predictions set to references reversed and punctuation removed.
    reversed_predictions = Predictions({"values": [
        "ertnec ytic eht ni gniK regruB eht raen si dna yldneirf ylimaf ton si mutnemilA",
        "yldneirf ylimaf ton si taht mutnemilA ertnec ytic eht ni ecalp si erehT",
        "snaelrO weN ni esuoh si erehT"
    ], "language": "en"})
