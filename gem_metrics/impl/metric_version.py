import subprocess
import json
import os
from pathlib import Path


def get_metric_hash():
    metric_list = [
        "bertscore",
        "bleu",
        "bleurt",
        "chrf",
        "local_recall",
        "meteor",
        "nist",
        "rouge",
        "msttr",
        "ngrams",
        "sari",
        "nubia",
        "questeval",
        "prism",
        "ter",
        "ttr",
        "yules_i",
        "wer",
        "cider",
        "moverscore",
    ]
    metric_version = {}
    for metric in metric_list:
        path = Path(os.path.abspath(os.pardir)) / metric
        metric_version[metric] = subprocess.check_output(
            f"git log -n 1 --pretty=format:%H -- {path}.py",
            shell=True,
            universal_newlines=True,
        )
    with open("metric_version.json", "w") as fp:
        json.dump(metric_version, fp, indent=2)


if __name__ == "__main__":
    get_metric_hash()
