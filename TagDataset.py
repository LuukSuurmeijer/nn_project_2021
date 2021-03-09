
#adapted from https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

from __future__ import absolute_import, division, print_function
import csv
import os
import datasets
import string


_DESCRIPTION = """\
This new dataset is designed for a POS tagger as intended for the Neural Networks 2021 course.
"""


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class TagDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "index":  datasets.Value("int32"),
                "word": datasets.Value("string"),
                "tag": datasets.Value("string")
                # These are the features of your dataset like images, labels ...
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": os.path.join(self.config.data_dir, "train.tsv"),
                "split": "train",},
        ),
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": os.path.join(self.config.data_dir, "test.tsv"),
                "split": "test"},
        ),
    ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset

        with open(filepath, encoding="utf-8") as f:
            data = csv.reader(f, delimiter='\t')
            for id_, row in enumerate(data):
                if len(row) == 3: #end of sentences character is skipped, see else
                    yield id_, {
                        "index": row[0],
                        "word": row[1].lower().translate(str.maketrans('', '', string.punctuation)),
                        "tag": row[2],
                        }
                else: #this is how i deal with the end of sentences asterisks, treat them as end of sentence tags with index -1
                    yield id_, {
                    "index": -1,
                    "word": row[0],
                    "tag": '<eos>',
                    }
