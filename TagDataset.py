
#adapted from https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

from __future__ import absolute_import, division, print_function
import csv
import os
import datasets
import string
from itertools import groupby


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
                "words": datasets.Sequence(datasets.Value("string")),
                "tags": datasets.Sequence(datasets.Value("string"))
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
            fulldata = [tuple(group) for key, group in groupby(data, lambda x: x[0] == '*') if not key]
            for id_, row in enumerate(fulldata):
                yield id_, {
                    "words": [item[1].lower() for item in row],
                    "tags": [item[2] for item in row],
                    }
