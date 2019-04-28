import sys, re, os
import argparse

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, "lib/InaNLP.jar"))

from IndonesianNLP import IndonesianSentenceFormalization
from IndonesianNLP import IndonesianSentenceDetector
from IndonesianNLP import IndonesianSentenceTokenizer

def formalize_sentence(sentence):
    formalizer = IndonesianSentenceFormalization()
    return formalizer.formalizeSentence(sentence)

def tokenize_sentence(sentence):
    tokenizer = IndonesianSentenceTokenizer()
    return tokenizer.tokenizeSentence(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    texts = []
    sentences_count = 0
    with open(args.infile, 'r') as input_file:
        for text in input_file:
            text = (re.sub('\.{2,}', '. ', text.rstrip()))
            text = (re.sub('\,+', ', ', text.rstrip()))
            text = (re.sub('!+', '! ', text.rstrip()))
            text = (re.sub('\?+', '? ', text.rstrip()))
            text = (re.sub('\(+', ' ( ', text.rstrip()))
            text = (re.sub('\)+', ' ) ', text.rstrip()))
            texts.append(text)

    with open(args.outfile, 'w') as output_file:
        for text in texts:
            output_file.write(formalize_sentence(text.lower()) + '\n')

    print("Number of processed text: " + str(len(texts)))
