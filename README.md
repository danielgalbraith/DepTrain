# DepTrain
Training dependency parser with spaCy in Python3.

Task to ensure rapper name 'Stefflon Don' tagged appropriately. Includes training sentences and test paragraph mentioning name. Uses standard spaCy English model, i.e. CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Command to run training:

`python train.py -m en [-n] [-o]`

Optional arguments for number of iterations [-n] and output directory [-o]. Visualizes dependency parse of test paragraph using displaCy, by default port 5000 (localhost:5000 in browser).

Repo contains source texts parsed by Stanford CoreNLP in .conllu format.

Requires spaCy 2.0+ and Python 3.0+.
