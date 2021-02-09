import unittest
from gem_metrics.bleu import BLEU
from gem_metrics.texts import Predictions, References


class NGramTest(unittest.TestCase):

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()
        self.references_identical = [
            {
                "target": ["Alimentum is not family-friendly, and is near the Burger King in the city centre."]
            },
            {
                "target": ["There is a place in the city centre, Alimentum, that is not family-friendly."]
            },
            {
                "target": ["There is a house in New Orleans."]
            }
        ]

        self.predictions = [
            "Alimentum is not family-friendly, and is near the Burger King in the city centre.",
            "There is a place in the city centre, Alimentum, that is not family-friendly.",
            "There is a house in New Orleans."
        ]

        self.bleu_metric = BLEU()

    def test_bleu_metric_identical_pred_ref(self):
        """Tests for identical predictions and references (BLEU = 100)
        """
        references = References({"values": self.references_identical, "language": "en"})
        predictions = Predictions({"values": self.predictions, "language": "en"})
        calculated_metrics = self.bleu_metric.compute(
            predictions=predictions, references=references)
        self.assertAlmostEqual(calculated_metrics["bleu"], 100.0)

    def test_bleu_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references (BLEU = 0.)
        """
        refs = [
            {
                "target": ["ertnec ytic eht ni gniK regruB eht raen si dna yldneirf ylimaf ton si mutnemilA"]
            },
            {
                "target": ["yldneirf ylimaf ton si taht mutnemilA ertnec ytic eht ni ecalp si erehT"]
            },
            {
                "target": ["snaelrO weN ni esuoh si erehT"]
            }
        ]
        predictions = Predictions({"values": self.predictions, "language": "en"})
        references = References({"values": refs, "language": "en"})
        calculated_metrics = self.bleu_metric.compute(
            predictions=predictions, references=references)
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.assertAlmostEqual(calculated_metrics["bleu"], 0.53, places=1)


    def test_bleu_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        tgt_empty = [
            "",
            "",
            ""
        ]
        predictions = Predictions({"values": tgt_empty, "language": "en"})
        references = References({"values": self.references_identical, "language": "en"})
        calculated_metrics = self.bleu_metric.compute(
            predictions=predictions, references=references)
        self.assertAlmostEqual(calculated_metrics["bleu"], 0)

if __name__ == '__main__':
    unittest.main()
