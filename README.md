# informationminer

Automatically performs NLP techniques.
Currently supports German and English language.

The following techniques are used on the passed text:
  - Tokenization
  - POS Tagging
    - English tagger is based on NLTK
    - German tagger is generated from TIGER corpus
  - Chunking
  - Named Entity recognition


## Install
Clone the package and use as module or run 'pip install informationminer'.

## Getting started

    from informationminer import InformationMiner
    iminer = InformationMiner("This is an example sentence.")
    print(iminer.ne)
