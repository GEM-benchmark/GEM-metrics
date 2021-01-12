#!/usr/bin/env python3

from argparse import ArgumentParser
import json
import sys

import gem_metrics


def main(args):
    """Main entry point -- load inputs, call metrics measuring, print outputs"""

    # load system predictions
    with open(args.predictions_file, encoding='UTF-8') as fh:
        data = json.load(fh)
    if isinstance(data, dict) and 'submission_name' in data:
        outs = gem_metrics.Submission(data)
    else:
        outs = gem_metrics.Predictions(data)

    # load references, if available
    if args.references_file is not None:
        refs = gem_metrics.References(args.references_file)
        assert(len(refs) == len(outs))

    values = gem_metrics.compute(outs, refs)

    # print output
    out_fh = sys.stdout
    if args.output_file != '-':
        out_fh = open(args.output_file, 'w', encoding='UTF-8')
    print(json.dumps(values, ensure_ascii=False, indent=4), file=out_fh)


if __name__ == '__main__':
    ap = ArgumentParser(description='GEM automatic metrics script')
    ap.add_argument('-o', '--output-file', type=str, help='Path to output file', default='-')
    ap.add_argument('-r', '--references-file', '--references', '--refs', type=str, help='Path to references JSON file')
    ap.add_argument('predictions_file', type=str, help='Path to system outputs JSON file')

    args = ap.parse_args()
    main(args)
