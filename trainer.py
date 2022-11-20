import torch
import random
import numpy as np
import os
from transformers import (
    AutoTokenizer
)
from torch.utils.data import TensorDataset
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

CACHE_DIR = "/cs/labs/oabend/shachar.don/pre-translationQE/clean_model/cache"


class Trainer:
    def __init__(
            self,
            model_type,
            model_name,
            num_labels=1,
            weight=None,
            args=None,
            use_cuda=True,
            output_dir=None,
            manual_seed=777,
            **kwargs,
    ):

        """
        Initializes a MonoTransQuest model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = {"manual_seed": manual_seed,
                     "labels_list": [0],
                     "output_dir": output_dir,
                     "do_lower_case": False,
                     "model_type": model_type,
                     "model_name": model_name,
                     "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
                     "no_cache": True,
                     "cache_dir": "/cs/labs/oabend/shachar.don/pre-translationQE/my_model/cache",
                     "max_seq_length": 128,
                     "reprocess_input_data": True,
                     'use_cached_eval_features': False,
                     'silent': False,
                     "multiprocessing_chunksize": 500,
                     "n_gpu": 1,
                     "sliding_window": False,
                     "stride": None,
                     "dataloader_num_workers": 1}

        self.num_labels = num_labels

        if self.args["manual_seed"]:
            random.seed(self.args["manual_seed"])
            np.random.seed(self.args["manual_seed"])
            torch.manual_seed(self.args["manual_seed"])
            if self.args["n_gpu"] and self.args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(self.args["manual_seed"])

        self.args["labels_list"] = [0]

        # update output_dir
        if output_dir:
            self.args["output_dir"] = output_dir

        print("model_name:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=self.args["do_lower_case"], cache_dir=CACHE_DIR, **kwargs
        )

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

    def load_and_cache_examples(
            self, examples, evaluate=False, no_cache=True,
            multi_label=False, verbose=True, silent=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args["process_count"]

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args["no_cache"]

        output_mode = "regression"

        if not no_cache:
            os.makedirs(self.args["cache_dir"], exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args["cache_dir"],
            "cached_{}_{}_{}_{}_{}".format(
                mode, args["model_type"], args["max_seq_length"],
                self.num_labels, len(examples),
            ),
        )
        print(cached_features_file)

        if os.path.exists(cached_features_file) and (
                (not args["reprocess_input_data"] and not no_cache)
                or (
                        mode == "dev" and args["use_cached_eval_features"] and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                print(
                    f" Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                print(
                    " Converting to features started. Cache is not used.")
                if args["sliding_window"]:
                    print(" Sliding window enabled")

            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in [
                    "xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(
                    args["model_type"] in ["roberta", "camembert",
                                        "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=
                tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in [
                    "xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args["silent"] or silent,
                stride=args["stride"],
                add_prefix_space=bool(
                    args["stride"] in ["roberta", "camembert",
                                        "xlmroberta", "longformer"]),
                # avoid padding in case of single example/online inferencing to decrease execution time
                pad_to_max_length=bool(len(examples) > 1),
                args=args,
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                       dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.float)

        all_label_types = torch.tensor([f.label_type for f in features],
                                       dtype=torch.int64)

        all_comet = torch.tensor([f.comet for f in features],
                                 dtype=torch.float)

        all_features = torch.tensor([f.features for f in features],
                                    dtype=torch.float)

        all_bert_score = torch.tensor([f.bert_score for f in features],
                                      dtype=torch.float)

        all_hter = torch.tensor([f.hter for f in features],
                                dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_label_ids,
                                all_label_types,
                                all_comet, all_features, all_bert_score,
                                all_hter)

        return dataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None, label_type=0,
                 comet=0, features=0, bert_score=0, hter=0):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.label_type = label_type
        self.comet = comet
        self.features = features
        self.bert_score = bert_score
        self.hter = hter


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 label_type=0, comet=0, features=0, bert_score=0, hter=0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_type = label_type
        self.comet = comet
        self.features = features
        self.bert_score = bert_score
        self.hter = hter


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end=False,
        sep_token_extra=False,
        pad_on_left=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        multi_label=False,
        silent=False,
        stride=None,
        add_prefix_space=False,
        pad_to_max_length=True,
        args=None,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            multi_label,
            stride,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]

    with Pool(process_count) as p:
        features = list(
            tqdm(
                p.imap(convert_example_to_feature, examples, chunksize=args["multiprocessing_chunksize"]),
                total=len(examples),
                disable=silent,
            )
        )
    return features


def convert_example_to_feature(
        example_row,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None  # we don't use the translation output.

    if example.text_b:
        if add_prefix_space and not example.text_b.startswith(" "):
            tokens_b = tokenizer.tokenize(" " + example.text_b)
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
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

    if tokens_b:
        if sep_token_extra:
            tokens += [sep_token]
            segment_ids += [sequence_b_segment_id]

        tokens += tokens_b + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if pad_to_max_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    # if output_mode == "classification":
    #     label_id = label_map[example.label]
    # elif output_mode == "regression":
    #     label_id = float(example.label)
    # else:
    #     raise KeyError(output_mode)

    # if output_mode == "regression":
    #     label_id = float(example.label)

    return InputFeatures(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=example.label,
        label_type=example.label_type, comet=example.comet, features=example.features, bert_score=example.bert_score,
        hter=example.hter
    )