import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer
)
from shared_paths_local import CACHE_DIR
from sklearn.model_selection import train_test_split

# Set pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_annotated_file(path, index="index", chrf=True, comet=True, bertScore=True):
    indices = []
    originals = []
    translations = []
    refs = []
    chrfs = []
    comets = []
    bertScores = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            refs.append(row["ref"])

            if comet:
                comets.append(float(row["comet"]))
            if bertScore:
                bertScores.append(float(row["bertScore"]))
            if chrf:
                chrfs.append(float(row["z_mean"]))

    df = pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         'ref': refs
         })

    if chrf:
        df["chrf"] = chrfs  # override the source sentence.

    if comet:
        df['comet'] = comets

    if bertScore:
        df['bertScore'] = bertScores

    return df


def convert_sent(sent, tokenizer):
    max_seq_length = 128
    sep_token = tokenizer.sep_token
    sequence_a_segment_id = 0
    cls_token = tokenizer.cls_token
    cls_token_segment_id = 0
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    mask_padding_with_zero = True
    pad_token_segment_id = 0

    if not sent.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + sent)
    else:
        tokens_a = tokenizer.tokenize(sent)

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 2
    if len(tokens_a) > max_seq_length - special_tokens_count:
        tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


class SystemsDataset(Dataset):
    def __init__(self, dfs, score_types, multilingual=False):

        self.num_system = len(dfs)
        self.use_chrf = True if "chrf" in score_types else False
        self.use_comet = True if "comet" in score_types else False
        self.use_bertScore = True if "bertScore" in score_types else False

        self.dfs = dfs
        # for i in range(self.num_system):
        #     df = read_annotated_file(paths_list[i], chrf=self.use_chrf, comet=self.use_comet,
        #                              bertScore=self.use_bertScore)
        #     self.dfs.append(df)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-large" if multilingual else "roberta-large", do_lower_case=False, cache_dir=CACHE_DIR)

    def __len__(self):
        return len(self.dfs[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx:", idx)

        sample = {}
        for i in range(self.num_system):
            df = self.dfs[i]
            # print("df:", df)
            row = df.iloc[idx]
            # print("row:", row)
            if i == 0:
                text = str(row["original"])
                # sample["original"] = text
                text = " ".join(text.split())

                inputs = self.tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=128,
                    truncation=True,
                    padding='max_length',
                    return_token_type_ids=True
                )
                ids = inputs['input_ids']
                mask = inputs['attention_mask']
                token_type_ids = inputs["token_type_ids"]

                # ids, mask, token_type_ids = convert_sent(str(row["original"]), self.tokenizer)

                sample["input_ids"] = torch.tensor(ids, dtype=torch.long)
                sample["input_mask"] = torch.tensor(mask, dtype=torch.long)
                sample["segment_ids"] = torch.tensor(token_type_ids, dtype=torch.long)

            if self.use_chrf:
                sample["chrf_%d" % i] = row["chrf"]
            if self.use_comet:
                sample["comet_%d" % i] = row["comet"]
            if self.use_bertScore:
                sample["bertScore_%d" % i] = row["bertScore"]

        if self.use_chrf:
            max_sys = np.argmax([sample["chrf_%d" % i] for i in range(self.num_system)])
            sample["best_chrf"] = max_sys
        elif self.use_comet:
            all_comet = [sample["comet_%d" % i] for i in range(self.num_system)]
            sample["all_comet"] = torch.as_tensor(all_comet, dtype=torch.float)
            max_sys = np.argmax(all_comet)
            sample["best_comet"] = max_sys
            # sample["best_chrf"] = np.zeros(self.num_system)
            # sample["best_chrf"][max_sys] = 1
            # print(sample["best_chrf"])
        # if self.use_comet:
        #     sample["best_comet"] = row["comet"]
        # if self.use_bertScore:
        #     sample["best_bertScore"] = row["bertScore"]
        # print(sample)
        return sample

