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

    # multi-file submissions
    if isinstance(data, dict) and 'submission_name' in data:
        data = gem_metrics.Submission(data)

        ref_data = None
        if args.references_file:
            with open(args.references_file, encoding='UTF-8') as fh:
                raw_ref_data = json.load(fh)
                assert(sorted(list(raw_ref_data.keys())) == sorted(data.datasets))
                for dataset in data.datasets:
                    ref_data[dataset] = gem_metrics.References(ref_data[dataset])
        values = {}
        for dataset in data.datasets:
            outs = data.predictions_for(dataset)
            # use default reference files if no custom ones are provided
            refs = ref_data[dataset] if ref_data else gem_metrics.load_references(dataset)
            if refs:
                assert(len(refs) == len(outs))
            values[dataset] = gem_metrics.compute(outs, refs)

    # single-file mode
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
