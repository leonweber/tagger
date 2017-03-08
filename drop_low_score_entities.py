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
confidence_threshold = np.percentile(confidences, args.percent_dropped)

def write_to_conll():
    with open(args.out, 'w') as f_out, open(args.tagged) as f_tagged,\
            open(args.true) as f_true:
        for tagged_line, true_line in zip(f_tagged, f_true):
            try:
                true_tag, token, pos_tag = true_line.split()
            except ValueError:
                f_out.write('\n')
                continue

            tagged_line =  tagged_line.split('__')
            pred_tag = 'O'

            if len(tagged_line) >= 3:
                confidence = float(tagged_line[-1])
                if confidence > confidence_threshold:
                    pred_tag = tagged_line[-2]

            f_out.write('{} {} {} {}\n'.format(token, pos_tag, true_tag, pred_tag))

write_to_conll()








