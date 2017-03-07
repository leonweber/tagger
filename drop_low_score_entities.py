#!/usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tagged', type=str, required=True)
parser.add_argument('--true', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--percent_dropped', type=int, required=True)

args = parser.parse_args()

def collect_confidences():
    confidences = []
    with open(args.tagged) as f:
        for line in f:
            line = line.split('__')
            if len(line) < 3:
                continue

            confidences.append(float(line[-1]))
    return np.array(confidences)

confidences = collect_confidences()
lowest_confidence_allowed = np.percentile(confidences, args.percent_dropped)

def write_to_conll():
    with open(args.out, 'w') as f_out, open(args.tagged) as f_tagged,\
            open(args.true) as f_true:
        for tagged_line, true_line in zip(f_tagged, f_true):
            confidence = None
            try:
                true_tag, token, pos_tag = true_line.split()
            except ValueError:
                f_out.write('\n')
                continue

            try:
                pred_token, pred_tag, confidence = tagged_line.split('__')
                confidence = float(confidence)
            except ValueError:
                pred_token, pred_tag = tagged_line.split('__')

            assert token == pred_token

            if confidence is None or confidence < lowest_confidence_allowed:
                pred_tag = 'O'

            f_out.write('{} {} {} {}\n'.format(token, pos_tag, true_tag, pred_tag))

write_to_conll()








