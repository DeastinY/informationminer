# informationminer

Automatically performs NLP techniques.
Currently supports German and English language.

The following techniques are used on the passed text:
  - Tokenization
  - POS Tagging
    - English tagger is based on NLTK **default**
    - German tagger is generated from TIGER corpus
  - Chunking
  - Named Entity recognition


## Install
This package is on pip ! Just use `pip3 install informationminer`.

## Getting started
Look at the following example. More complex tasks like creating your own Tagger will be added later.
```python
>>> import informationminer
>>> im = informationminer.InformationMiner("This is a sample sentence.")
INFO:root:Start processing text
INFO:root:Tokenizing text
INFO:root:Creating new tokens
INFO:root:Writing output/01_token_output.json
INFO:root:POS tagging tokens
INFO:root:Creating new POS tags. This can take some time ...
WARNING:root:No Tagger found. Generating Tagger, this may take some time.
INFO:root:Reading TIGER corpus
INFO:root:Shuffling sentences
INFO:root:Training Tagger
INFO:root:Training finished
INFO:root:Serializing the Tagger
INFO:root:Writing output/02_pos_output.json
INFO:root:Chunking POS
INFO:root:Creating new chunks. This can take some time ...
INFO:root:Writing output/03_chunk_output.pickle
INFO:root:Extracting entity names
INFO:root:Searching for named entities
INFO:root:Writing output/04_ne_output.json
INFO:root:Processing finished in 35.50 s   
>>> im.tokens
['This', 'is', 'a', 'sample', 'sentence', '.']
>>> im.pos
[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sample', 'JJ'), ('sentence', 'NN'), ('.', '.')]

```

The InformationMiner class has a couple of optional parameters:
    - **save_output**: Write output to outdir/outfile. Enabled by default so you don't do work twice.
    - **force_create**: Will allways overwrite files if *save_output* is enabled
    - **language**: German by default. Either *ger* or *en*
