import os
import json
import nltk
import numpy
from progress.bar import Bar
from pathlib import Path
import pickle
import logging
import textract
import time
from .POSTagger import tag
from sklearn import decomposition
import sklearn.feature_extraction.text as sklearn_text

logging.basicConfig(level=logging.DEBUG)


class InformationMiner:
    def __init__(self, text, save_output=False, language="english", outdir="output", outfile="output", force_create=False):
        """
        Creates an instance of InformationMiner.
        :param text: The text to process. Either a single string, or a list of strings.
        :param save_output: True if the output should be saved to oudir. Will create several files.
        :param language: Either german or english.
        :param outdir: The directory to output files to
        :param outfile: The name of the output file. Different steps will add prefixes
        :param force_create: If set to True will overwrite any output.
        """
        self.save_output = save_output
        self.language = language
        self.outdir = Path(outdir)
        if not self.outdir.exists:
            os.makedirs(outdir)
        self.outfile = outfile
        self.force_create = force_create
        self.tokens = None
        self.pos = None
        self.chunk = None
        self.ne = None
        self.topics = None
        self.text = text if not isinstance(text, str) else [text]

    def process(self, text=None):
        logging.info("Start processing text")
        start = time.time()
        self.text = text if text else self.text
        self.tokens = self.tokenize()
        if self.language == "german":
            self.pos = self.tag_pos_ger()
        elif self.language == "english":
            self.pos = self.tag_pos_en()
        else:
            msg = "Language should be either english or german"
            logging.error(msg)
            raise Exception(msg)
        self.chunk = self.ne_chunk()
        self.ne = self.extract_entity_names()
        self.topics = self.nmf()
        stop = time.time()
        logging.info("Processing finished in {:.2f} s".format(stop - start))

    def tokenize(self):
        return self.exec_cached_func("Tokenizing text",
                                     "Creating new tokens",
                                     self.text,
                                     '01_token_',
                                     lambda d: nltk.word_tokenize(d, self.language),
                                     False)

    def ne_chunk(self):
        return self.exec_cached_func("Chunking POS",
                                     "Creating new chunks. This can take some time ...",
                                     self.pos,
                                     '03_chunk_',
                                     lambda d: nltk.ne_chunk(d),
                                     True)

    def tag_pos_ger(self):
        return self.exec_cached_func("POS tagging tokens",
                                     "Creating new POS tags. This can take some time ...",
                                     self.tokens,
                                     '02_pos_',
                                     lambda d: tag(d),
                                     False)

    def tag_pos_en(self):
        return self.exec_cached_func("POS tagging tokens",
                                     "Creating new POS tags. This can take some time ...",
                                     self.tokens,
                                     '02_pos_',
                                     lambda d: nltk.pos_tag(d),
                                     False)

    def extract_entity_names(self):
        return self.exec_cached_func("Extracting entity names",
                                     "Searching for named entities",
                                     self.chunk,
                                     '04_ne_',
                                     lambda d: self.extract_recurse(d),
                                     False)

    def nmf(self):
        return self.exec_cached_func("Extracting topics",
                                     "Generating topics",
                                     self.text,
                                     '05_nmf',
                                     lambda d: self.nonnegative_matrix_factorization(d),
                                     False)

    def nonnegative_matrix_factorization(self, text):
        vectorizer = sklearn_text.CountVectorizer(input='text', stop_words=self.language, min_df=20)
        dtm = vectorizer.fit_transform(text)
        vocab = vectorizer.get_feature_names()
        num_topics, num_top_words = 20, 20
        clf = decomposition.NMF(n_components=num_topics, random_state=1)
        clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
            word_idx = numpy.argsort(topic)[::-1][0:num_top_words]
            topic_words.append([vocab[i] for i in word_idx])
        return topic_words

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
        return Path(self.outdir / (prefix + self.outfile + '.pickle' if binary else '.json'))

    def save(self, data, prefix='', binary=False):
        if not self.save_output:
            return
        file = self.get_file(prefix, binary)
        if file.exists() and not self.force_create:
            logging.warning("Did not write {}. Already exists and overwrite is disabled.".format(file))
        else:
            logging.debug("Writing {}".format(file))
            if binary:
                with file.open('wb') as fout:
                    pickle.dump(data, fout, protocol=3)
            else:
                with file.open('w') as fout:
                    json.dump(data, fout)

    def get_cached(self, prefix, binary):
        file = self.get_file(prefix, binary)
        if file.exists() and not self.force_create:
            if binary:
                logging.info("Loading cached file")
                with file.open('rb') as fin:
                    return pickle.load(fin)
            else:
                logging.info("Loading cached file")
                with file.open('r') as fin:
                    return json.load(fin)

    def exec_cached_func(self, log_msg, log_msg_create, data, prefix, func, binary):
        logging.debug(log_msg)
        cached = self.get_cached(prefix, binary)
        if not cached:
            logging.debug(log_msg_create)
            bar = Bar(log_msg_create, max = len(data))
            res = []
            for d in data:
                bar.next()
                res.append(func(d))
            bar.finish()
            self.save(res, prefix, binary)
        return cached if cached else res


if __name__ == '__main__':
    def get_text():
        infile = Path('input.txt')
        if infile.exists():
            logging.debug("Reading file from disk.")
            with infile.open('r') as fin:
                text = fin.readlines()
        else:
            logging.debug("Creating new file from PDF.")
            text = textract.process(
                '/home/ric/Nextcloud/rpg/shadowrun/rulebooks/Shadowrun_5_Grundregelwerk.pdf').decode('utf-8')
            with infile.open('w') as fout:
                fout.writelines(text)
        return text


    InformationMiner("\n".join(get_text()), force_create=True)
    # InformationMiner("Peter ist ein großer Junge. Er kauft bei dem großen Supermarkt Tedi schon ganz alleine eine Frisbee.", outfile='short_test', force_create=True)
