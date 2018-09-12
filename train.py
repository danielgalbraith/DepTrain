#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy import displacy

# training data
TRAIN_DATA = [
    ("Stefflon Don is on the periphery of global greatness.", {
        'deps': ['compound', 'nsubj', 'cop', 'case', 'det', 'root', 'case', 'amod', 'nmod', 'punct']
    }),
    ("From Jools Holland to the BBC Sound Poll, Steff's powerful presence commands attention both on record and in real life.", {
        'deps': ['case', 'compound', 'nmod', 'case', 'det', 'compound', 'compound', 'nmod', 'punct', 'nmod:poss', 'case', 'amod', 'nsubj', 'root', 'dobj', 'cc:preconj', 'case', 'nmod', 'cc', 'case', 'amod', 'conj', 'punct']
    })
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=10):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe('parser')

    # add labels to the parser
    for _, annotations in TRAIN_DATA:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, losses=losses)
            print(losses)

    # test the trained model
    test_text = "It was back in 2007 that hip-hop bible XXL launched its first ever Freshman Class, a list of ten up-and-coming artists poised to change the rap game for good. The last decade has seen more than a hundred stars spotlighted as part of the list and its accompanying annual cover feature, but this year features a history-making entry: Stefflon Don. The talented star has already built a strong reputation for herself in the UK; her unique blend of hard-hitting raps and smooth, dancehall beats has galvanized the scene, earning her critical acclaim and a series of impressive chart positions. Now, she seems ready to achieve the unthinkable: global stardom. Earlier this year, her infectious hit “Hurtin’ Me” – featuring former XXL Freshman French Montana – ascended the Billboard charts, peaking at no. 7 and confirming her US fanbase; but could she truly become the first artist to crack the US? And, more importantly, why has it taken so long for UK rappers to achieve Stateside success?"
    doc = nlp(test_text)
    print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])
    sentence_spans = list(doc.sents)
    displacy.serve(sentence_spans, style='dep')

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])

if __name__ == '__main__':
    plac.call(main)
