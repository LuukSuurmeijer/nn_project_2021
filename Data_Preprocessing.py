import conllu
import re
from collections import Counter

# clean the file because conllu.parse function needs id as first column, "_" and not "-" as None values and no floats in 'head' column
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
                line = line.replace(" - "," _ ")
                line = re.sub(r"([0-9]+)\.([0-9]+)", "_", line)
            fout.write(line)

# returns a list pf TokenLists, each TokenList is a sentence
# each item in a token list has a corresponding dictionary storing information e.g.
# {'id': 0, 'form': 'Xinhua', 'lemma': 'NNP', 'upos': '(TOP(FRAG(NP*', 'xpos': None, 'feats': None, 'head': None, 'deprel': '_', 'deps': '(ORG*', 'misc': None}
# in our case keys are not named correctly because the columns do not have the correct order (e.g. the key "lemma" actually stores the POS tag)
# for more information see: https://pypi.org/project/conllu/
def parse_conll_file(file):
    with open(file) as f:
        data = f.read()

    sentences = conllu.parse(data)
    # have to drop the last element of the sentence list because it is for whatever reason just an empty TokenList
    sentences.pop(-1)
    print(sentences[0][0])
    return sentences


def get_number_of_sentences(sentences):
    return len(sentences)

# get a list which stores at index 0 the length of sentence 0 and so on
def get_sentence_lengths(sentences):
    sentence_lengths = []
    for sentence in sentences:
        sentence_length = len(sentence)
        sentence_lengths.append(sentence_length)
    return sentence_lengths


def get_sentence_length_stats(sentence_lengths):
    average_sen_len = sum(sentence_lengths) / len(sentence_lengths)
    return max(sentence_lengths), min(sentence_lengths), average_sen_len


def create_tab_file(sentences, outfile):
    with open(outfile, "w+") as f:
        for sentence in sentences:
            for word in sentence:
                f.write(f"{word['id']}\t{word['form']}\t{word['lemma']}\n")
            f.write("*\n")


def get_tag_statistics(sentences):
    list_of_tags = []
    for sentence in sentences:
        for word in sentence:
            list_of_tags.append(word['lemma'])
    c = Counter(list_of_tags)
    number_of_tags = len(list_of_tags)
    control_sum = 0
    for tag in Counter(list_of_tags).keys():
        control_sum += round(c[tag]/number_of_tags*100,2)
        print(f"{tag}\t{round(c[tag]/number_of_tags*100,2)}%")

def print_information_about_data(sentences):
    sentence_lengths = get_sentence_lengths(sentences)
    min, max, av, = get_sentence_length_stats(sentence_lengths)
    print(f"Maximum sequence length: {min}\n"
          f"Minimum sequence length: {max}\n"
          f"Mean sequence length: {av}\n"
          f"Number of sentences: {get_number_of_sentences(sentences)}\n")
    print("Tags:")
    get_tag_statistics(sentences)


def main():
    infile = "sample.conll"
    outfile = "cleaned.conll"
    clean_file(infile,outfile)
    sentences = parse_conll_file(outfile)
    create_tab_file(sentences, "sample.tsv")
    print_information_about_data(sentences)

if __name__ == '__main__':
    main()