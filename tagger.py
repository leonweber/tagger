#!/usr/bin/env python

import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, zero_digits
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)


def pred_to_alpha_masks(y_pred, label_to_index, index_to_label, epsilon=1e-7):
    """
    y_pred $\in \{0, .., n_tags-1\}^{seq_len} are the predicted tags for one example
    We assume that only valid tag-transitions are predicted
    """

    masks = []
    prev_label = None
    i_label_index = None
    mask = None
    for seq_idx, label_idx in enumerate(y_pred):
        label = index_to_label[label_idx]

        # enforce negative constraints
        if prev_label is not None and prev_label != 'O' and (label == 'O' or label.startswith('B') or label.startswith('S')):
            mask[seq_idx, i_label_index] = 0
            masks.append(mask + epsilon)
            mask = None

        if label.startswith('B') or label.startswith('S'):
            mask = np.ones((len(y_pred)+2, len(label_to_index)+2), dtype='float32')
            relevant_b_label = label
            i_label_index = label_to_index['I' + relevant_b_label[1:]]

        # enforce positive constraints
        if label != 'O':
            mask[seq_idx, :] = 0
            mask[seq_idx, label_idx] = 1

        prev_label = label

    if mask is not None:
        masks.append(mask + epsilon)

    return masks




# Load existing model
print "Loading model..."
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval, f_conf = model.build(training=False, **parameters)
model.reload()

f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

print 'Tagging...'
with codecs.open(opts.input, 'r', 'utf-8') as f_input:
    count = 0
    for line in f_input:
        words = line.rstrip().split()
        if line:
            count += 1
            # Lowercase sentence
            if parameters['lower']:
                line = line.lower()
            # Replace all digits with zeros
            if parameters['zeros']:
                line = zero_digits(line)
            # Prepare input
            sentence = prepare_sentence(words, word_to_id, char_to_id,
                                        lower=parameters['lower'])
            input = create_input(sentence, parameters, False)
            # Decoding
            if parameters['crf']:
                y_preds = np.array(f_eval(*input))[1:-1]
                alpha_masks = pred_to_alpha_masks(y_preds, {v: k for k, v in model.id_to_tag.items()}, model.id_to_tag)
                confidences_per_entity = []
                for alpha_mask in alpha_masks:
                    conf_input = input + [alpha_mask]
                    conf = np.array(f_conf(*conf_input))
                    confidences_per_entity.append(conf)
            else:
                y_preds = f_eval(*input).argmax(axis=1)
            y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
            # Output tags in the IOB2 format
            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)
            # Write tags
            assert len(y_preds) == len(words)
            for word, tag in zip(words, y_preds):
                if tag.startswith('B'):
                    # try:
                    confidence = confidences_per_entity.pop(0)
                if tag == 'O':
                    confidence = None
                if confidence is not None:
                    f_output.write('%s%s%s%s%s\n' % (word, opts.delimiter, tag, opts.delimiter, confidence))
                else:
                    f_output.write('%s%s%s\n' % (word, opts.delimiter, tag))
            f_output.write('\n')
        else:
            f_output.write('\n')
        if count % 100 == 0:
            print count

print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
f_output.close()
