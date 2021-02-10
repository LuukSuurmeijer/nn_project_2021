import conllu
import re
import pandas as pd


def clean_file(infile,outfile):
    delete_list = ["nw/xinhua/02/chtb_0223   0   ", "nw/xinhua/02/chtb_0223   0    ", "bc/phoenix/00/phoenix_0001   0    ",
                   "bc/phoenix/00/phoenix_0001   0   ", "bc/phoenix/00/phoenix_0001   1    ",
                   "bc/phoenix/00/phoenix_0001   1   ", "bc/phoenix/00/phoenix_0001   2    ", "bc/phoenix/00/phoenix_0001   2   ",
                   "bc/phoenix/00/phoenix_0001   3    ", "bc/phoenix/00/phoenix_0001   3   ", "bc/phoenix/00/phoenix_0001   4    ",
                   "bc/phoenix/00/phoenix_0001   4   ", "bc/phoenix/00/phoenix_0001   5    ", "bc/phoenix/00/phoenix_0001   5   ",
                   "bc/phoenix/00/phoenix_0001   6    ", "bc/phoenix/00/phoenix_0001   6   ", "bn/pri/00/pri_0016   0    ",
                   "bn/pri/00/pri_0016   0   ", "nw/wsj/16/wsj_1681   0    ", "nw/wsj/16/wsj_1681   0   "]

    with open(infile) as fin, open(outfile, "w+") as fout:
        for line in fin:
            for word in delete_list:
                line = line.replace(word, "")
                line = line.replace("-","_")
                line = re.sub(r"([0-9]+)\.([0-9]+)", "_", line)
            fout.write(line)


def parse_conll_file(file):
    with open(file) as f:
        data = f.read()

    sentences = conllu.parse(data)
    # have to drop the last element of the sentence list because it is for whatever reason just an empty TokenList
    sentences.pop(-1)
    return sentences


def get_number_of_sentences(sentences):
    return len(sentences)


def get_sentence_lengths(sentences):
    sentence_lengths = []
    for sentence in sentences:
        sentence_length = len(sentence)
        sentence_lengths.append(sentence_length)
    return sentence_lengths


def get_sentence_length_stats(sentence_lengths):
    average_sen_len = sum(sentence_lengths) / len(sentence_lengths)
    return max(sentence_lengths), min(sentence_lengths), average_sen_len

# data = pd.read_csv("sample.conll", sep="\t", names=["document name", "document id", "id", "word", "POS"], comment="#")
# print(data)

def main():
    infile = "sample.conll"
    outfile = "cleaned.conll"
    clean_file(infile,outfile)
    sentences = parse_conll_file(outfile)
    sentence_lengths = get_sentence_lengths(sentences)
    length_stats = get_sentence_length_stats(sentence_lengths)
    print(length_stats)


if __name__ == '__main__':
    main()