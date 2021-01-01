#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import Optional
import json
import sys

import gem_metrics


def compute_metrics(outs: gem_metrics.Predictions, refs: Optional[gem_metrics.References]) -> dict:
    """Main metrics computation routine. Expects a Predictions and a References object, holding
    system outputs and corresponding references (References may be None -- only referenceless metrics
    are computed in such a case).
    Returns a dict with the results.
    """
    # initialize values storage
    values = {'predictions_file': outs.filename,
              'N': len(outs)}

    # compute referenceless metrics
    for metric_class in gem_metrics.REFERENCELESS_METRICS:
        metric = metric_class()
        values.update(metric.compute(outs))

    # compute ref-based metrics
    if args.references_file is not None:
        values['references_file'] = refs.filename
        for metric_class in gem_metrics.REFERENCED_METRICS:
            metric = metric_class()
            values.update(metric.compute(outs, refs))
    return values


def main(args):
    """Main entry point -- load inputs, call metrics measuring, print outputs"""

    # load system predictions
    outs = gem_metrics.Predictions(args.predictions_file)

    # load references, if available
    if args.references_file is not None:
        refs = gem_metrics.References(args.references_file)
        assert(len(refs) == len(outs))

    values = compute_metrics(outs, refs)

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
