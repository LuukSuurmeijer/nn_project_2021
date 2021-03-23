import argparse
import os
from collections import Counter

def get_sentences(infile, outfile):
    with open(infile, encoding='UTF-8') as fin, open(outfile, "w+", encoding='UTF-8') as fout:
        tags = Counter()
        sentences = []
        sentence = []
        for line in fin:
            if line[0] != '#' and line != '\n':
                index = line.split()[2]
                word = line.split()[3]
                POS = line.split()[4]
                if word in ['"', "'"]:
                    word = '\\'+word
                fout.write(index + '\t' + word + '\t' + POS + '\n')
                tags[POS] += 1
                sentence.append(word)
            if line == '\n':
                fout.write('*\n')
                sentences.append(sentence)
                sentence = []
    return tags, sentences


def get_info(tags, sentences, outfile):
    with open(outfile, "w+") as fout:
        fout.write(f'Number of sentences: {len(sentences)}\n')
        sentence_length = [len(elem) for elem in sentences]
        max_sent = max(sentence_length)
        min_sent = min(sentence_length)
        avg_sent = sum(sentence_length) / len(sentences)

        fout.write(f'Maximum sequence length: {max_sent}\n')
        fout.write(f'Minimum sequence length: {min_sent}\n')
        fout.write(f'Average sequence length: {avg_sent}\n')

        fout.write('\nTags\n')
        tags_total = sum(tags.values())
        for k, v in tags.items():
            percentage = v * 100.0 / tags_total
            fout.write(f'{k}\t{round(percentage,2)}%\n')


def main():
    parser = argparse.ArgumentParser(description='Preprocess the conll data.')
    parser.add_argument('input_file', help="input file (concatenated conll file)")
    parser.add_argument('output_dir', help="output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tsv_file = "data.tsv"
    info_file = "data.info"

    tags, sentences = get_sentences(args.input_file, args.output_dir+tsv_file)
    get_info(tags, sentences, args.output_dir+info_file)


if __name__ == '__main__':
    main()
