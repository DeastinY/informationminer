import os
import json
import nltk
import numpy
from progress.bar import Bar
import pickle
import logging
import textract
import time
from .POSTagger import tag

logging.basicConfig(level=logging.DEBUG)


class InformationMiner:
    def __init__(self, text, save_output=False, language="en", outdir="output", outfile="output", force_create=False):
        """
        Creates an instance of InformationMiner.
        :param text: The text to process. Either a single string, or a list of strings.
        :param save_output: True if the output should be saved to oudir. Will create several files.
        :param language: Either ger or en.
        :param outdir: The directory to output files to
        :param outfile: The name of the output file. Different steps will add prefixes
        :param force_create: If set to True will overwrite any output.
        """
        self.save_output = save_output
        self.language = language
        self.outdir = outdir
        if not os.path.exists(outdir):
                os.makedirs(outdir)
        self.outfile = outfile
        self.force_create = force_create
        self.tokens = None
        self.pos = None
        self.chunk = None
        self.ne = None
        self.text = text if not isinstance(text, str) else [text]
        self.process()

    def process(self, text=None):
        logging.debug("Start processing text")
        start = time.time()
        self.text = text if text else self.text
        self.tokens = self.tokenize()
        if self.language == "ger":
            self.pos = self.tag_pos_ger()
        elif self.language == "en":
            self.pos = self.tag_pos_en()
        else:
            raise Exception("Language should be either en or ger")
        self.chunk = self.ne_chunk()
        self.ne = self.extract_entity_names()
        stop = time.time()
        logging.debug("Processing finished in {:.2f} s".format(stop - start))

    def tokenize(self):
        return self.exec_cached_func("Tokenizing text",
                                     "Creating new tokens",
                                     self.text,
                                     '01_token_',
                                     lambda d: [nltk.word_tokenize(i, 'german') for i in d],
                                     False)

    def ne_chunk(self):
        return self.exec_cached_func("Chunking POS",
                                     "Creating new chunks. This can take some time ...",
                                     self.pos,
                                     '03_chunk_',
                                     lambda d: [nltk.ne_chunk(i) for i in d],
                                     True)

    def tag_pos_ger(self):
        return self.exec_cached_func("POS tagging tokens",
                                     "Creating new POS tags. This can take some time ...",
                                     self.tokens,
                                     '02_pos_',
                                     lambda d: [tag(i) for i in d],
                                     False)

    def tag_pos_en(self):
        return self.exec_cached_func("POS tagging tokens",
                                     "Creating new POS tags. This can take some time ...",
                                     self.tokens,
                                     '02_pos_',
                                     lambda d: [nltk.pos_tag(i) for i in d],
                                     False)

    def extract_entity_names(self):
        return self.exec_cached_func("Extracting entity names",
                                     "Searching for named entities",
                                     self.chunk,
                                     '04_ne_',
                                     lambda d: [self.extract_recurse(i) for i in d],
                                     False)

    def extract_recurse(self, tree):
        entity_names = []
        if hasattr(tree, 'label') and tree.label():
            if tree.label() != 'S':
                entity_names.append(' '.join([t[0] for t in tree]))
            for child in tree:
                entity_names.extend(self.extract_recurse(child))
        else:
            if 'NE' in tree[1]:
                entity_names.append(tree[0])
        return list(set(entity_names))

    #######################################Util Functions Down Here #######################################

    def get_file(self, prefix, binary=False):
        outfile = os.path.join(self.outdir, prefix + self.outfile)
        outfile += '.pickle' if binary else '.json'
        return outfile

    def save(self, data, prefix='', binary=False):
        if not self.save_output:
            return
        file = self.get_file(prefix, binary)
        if os.path.exists(file) and not self.force_create:
            logging.warning("Did not write {}. Already exists and overwrite is disabled.".format(file))
        else:
            logging.debug("Writing {}".format(file))
            if binary:
                with open(file, 'wb') as fout:
                    pickle.dump(data, fout, protocol=3)
            else:
                with open(file, 'w') as fout:
                    json.dump(data, fout)

    def get_cached(self, prefix, binary):
        file = self.get_file(prefix, binary)
        if os.path.exists(file) and not self.force_create:
            if binary:
                logging.info("Loading cached file")
                with open(file, 'rb') as fin:
                    return pickle.load(fin)
            else:
                logging.info("Loading cached file")
                with open(file, 'r') as fin:
                    return json.load(fin)

    def exec_cached_func(self, log_msg, log_msg_create, data, prefix, func, binary):
        logging.debug(log_msg)
        cached = self.get_cached(prefix, binary)
        if not cached:
            logging.debug(log_msg_create)
            res = func(data)
            self.save(res, prefix, binary)
        return cached if cached else res


if __name__ == '__main__':
    def get_text():
        infile = 'input.txt'
        if os.path.exists(infile):
            logging.debug("Reading file from disk.")
            with open(infile, 'r') as fin:
                text = fin.readlines()
        else:
            logging.debug("Creating new file from PDF.")
            text = textract.process(
                '/home/ric/Nextcloud/rpg/shadowrun/rulebooks/Shadowrun_5_Grundregelwerk.pdf').decode('utf-8')
            with open(infile, 'w') as fout:
                fout.writelines(text)
        return text


    InformationMiner("\n".join(get_text()))
    # InformationMiner("Peter ist ein großer Junge. Er kauft bei dem großen Supermarkt Tedi schon ganz alleine eine Frisbee.", outfile='short_test', force_create=True)
