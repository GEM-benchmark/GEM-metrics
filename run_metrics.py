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
    sources_file: str = ""
    output_file: str = ""
    use_heavy_metrics: bool = False
    metric_list: list = None


def main(config):
    """Main entry point -- load inputs, call metrics measuring, print outputs"""
    if config.use_heavy_metrics:
        config.metric_list.append('bertscore')
        config.metric_list.append('bleurt')
        config.metric_list.append('questeval')

    metric_dict = gem_metrics.metric_list_to_metric_dict(config.metric_list)

    # load system predictions
    with open(config.predictions_file, encoding='UTF-8') as fh:
        data = json.load(fh)

    # multi-file submissions
    if isinstance(data, dict) and 'submission_name' in data:
        data = gem_metrics.Submission(data)

        ref_data = None
        if config.references_file:
            with open(args.references_file, encoding='UTF-8') as fh:
                ref_data = json.load(fh)
                for dataset in ref_data.keys():
                    ref_data[dataset] = gem_metrics.References(ref_data[dataset])

        src_data = None
        if config.sources_file:
            with open(args.references_file, encoding='UTF-8') as fh:
                src_data = json.load(fh)
                for dataset in src_data.keys():
                    src_data[dataset] = gem_metrics.Sources(src_data[dataset])

        values = gem_metrics.process_submission(data, ref_data, src_data, metric_dict)

    # single-file mode
    else:
        outs = gem_metrics.Predictions(data)
        srcs = None
        refs = None

        # load references, if available
        if config.references_file is not None:
            refs = gem_metrics.References(args.references_file)
            assert(len(refs) == len(outs))

        # load sources, if available
        if config.sources_file is not None:
            srcs = gem_metrics.Sources(args.sources_file)
            assert(len(srcs) == len(outs))

        values = gem_metrics.compute(outs, refs, srcs, metric_dict)

    # print output
    out_fh = sys.stdout
    if config.output_file:
        out_fh = open(args.output_file, 'w', encoding='UTF-8')
    print(json.dumps(values, ensure_ascii=False, indent=4), file=out_fh)


if __name__ == '__main__':
    ap = ArgumentParser(description='GEM automatic metrics script')
    ap.add_argument('predictions_file', type=str, help='Path to system outputs JSON file')
    ap.add_argument('-r', '--references-file', '--references', '--refs', type=str, help='Path to references JSON file')
    ap.add_argument('-s', '--sources-file', '--sources', '--srcs', type=str, help='Path to sources JSON file')
    ap.add_argument('-o', '--output-file', type=str, help='Path to output file', default='')
    ap.add_argument('--heavy-metrics', action='store_true', help='Run heavyweight metrics (BERTScore, BLEURT and SAFEval)')
    ap.add_argument('--metric-list', nargs='+', default=['bleu', 'meteor', 'rouge', 'msttr', 'ngram', 'sari', 'local_recall'],
                    help=('Full metric list default is [bleu, meteor, rouge, msttr, ngram, sari, local_recall]. '
                          + 'You can add bertscore, bleurt and questeval by manually adding them in the command '
                          + 'line argument here, or by using the --heavy-metrics flag'))
    args = ap.parse_args()

    # Workaround for metrics that use cmd flags - write all args to config.
    config = Config(
        predictions_file=args.predictions_file,
        references_file=args.references_file,
        sources_file=args.sources_file,
        output_file=args.output_file,
        use_heavy_metrics=args.heavy_metrics,
        metric_list=args.metric_list,
    )

    sys.argv = sys.argv[:1]
    main(config)
