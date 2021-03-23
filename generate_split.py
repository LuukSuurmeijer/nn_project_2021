import csv
import argparse
import os

#i only want to split on whole sentences, so I look for the closest '*' based on split
def find_nearest_delimiter(index, array):
    for i in range(index,len(array)+1):
        if array[i][0] == '*':
            return i


def randomsplit(corpus, split, outdir=""):
    # example: split = 0.8
    # train = 80% of the data, test = 10%, valid = 10%
    trainsplit = find_nearest_delimiter(int(len(corpus) * split), corpus)
    train = corpus[0:trainsplit+1]
    test = corpus[trainsplit+1:]

    print(len(train), len(test))

    path = "train_test_split"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    with open(f"{outdir}/train.tsv", 'w', encoding='utf-8') as tr:
        writer = csv.writer(tr, delimiter='\t', lineterminator='\n')
        writer.writerows(train)
    with open(f"{outdir}/test.tsv", 'w', encoding='utf-8') as te:
        writer = csv.writer(te, delimiter='\t', lineterminator='\n')
        writer.writerows(test)


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, default='brown',help="filename/directory for test file")
parser.add_argument("--outdir", type=str, default="", help="directory for the processed corpus files")
parser.add_argument("--split", type=float, default=0.9, help="ratio of data to use for training, the rest will be used for validation and test (50/50)")
args = parser.parse_args()

with open(args.data, 'r') as f:
    corpus = list(csv.reader(f, delimiter='\t'))

randomsplit(corpus, args.split, args.outdir)
