#!/usr/bin/env python3

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import sys

import gem_metrics


@dataclass
class Config:
    predictions_file: str = ""
    references_file: str = ""
    output_file: str = ""


def main(config):
    """Main entry point -- load inputs, call metrics measuring, print outputs"""


    # load system predictions
    with open(config.predictions_file, encoding='UTF-8') as fh:
        data = json.load(fh)

    # multi-file submissions
    if isinstance(data, dict) and 'submission_name' in data:
        data = gem_metrics.Submission(data)

        ref_data = None
        if config.references_file:
            with open(args.references_file, encoding='UTF-8') as fh:
                raw_ref_data = json.load(fh)
                assert(sorted(list(raw_ref_data.keys())) == sorted(data.datasets))
                for dataset in data.datasets:
                    ref_data[dataset] = gem_metrics.References(ref_data[dataset])

        values = gem_metrics.process_submission(data, ref_data)

    # single-file mode
    else:
        outs = gem_metrics.Predictions(data)

        # load references, if available
        if config.references_file is not None:
            refs = gem_metrics.References(args.references_file)
            assert(len(refs) == len(outs))

        values = gem_metrics.compute(outs, refs)

    # print output
    out_fh = sys.stdout
    if config.output_file:
        out_fh = open(args.output_file, 'w', encoding='UTF-8')
    print(json.dumps(values, ensure_ascii=False, indent=4), file=out_fh)


if __name__ == '__main__':
    ap = ArgumentParser(description='GEM automatic metrics script')
    ap.add_argument('predictions_file', type=str, help='Path to system outputs JSON file')
    ap.add_argument('-r', '--references-file', '--references', '--refs', type=str, help='Path to references JSON file')
    ap.add_argument('-o', '--output-file', type=str, help='Path to output file', default='')
    args = ap.parse_args()

    # Workaround for metrics that use cmd flags - write all args to config.
    config = Config(
        predictions_file=args.predictions_file,
        references_file=args.references_file,
        output_file=args.output_file)
    sys.argv = sys.argv[:1]
    main(config)
