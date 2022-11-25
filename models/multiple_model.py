import os
import shutil
import sys
# sys.path.append("/cs/labs/oabend/shachar.don/pre-translationQE/clean_model/TransQuest/")
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,

)
import random
from trainer import InputExample
from shared_paths_local import CACHE_DIR


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class MultipleModel(nn.Module):

    def __init__(self, seed, multilingual=False):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        super(MultipleModel, self).__init__()
        model_name = "roberta-large"
        if multilingual:
            model_name = "xlm-" + model_name
        self.config = RobertaConfig.from_pretrained(model_name, cache_dir=CACHE_DIR, num_labels=1)
        self.roberta = RobertaModel.from_pretrained(model_name, cache_dir=CACHE_DIR, config=self.config)

        # CHRF
        self.chrf_pre_classifier = torch.nn.Linear(1024, 1024)
        self.chrf_dropout = torch.nn.Dropout(0.1)
        self.chrf_classifier = torch.nn.Linear(1024, 1)

        # DA
        self.da_pre_classifier = torch.nn.Linear(1024, 1024)
        self.da_dropout = torch.nn.Dropout(0.1)
        self.da_classifier = torch.nn.Linear(1024, 1)

        # COMET
        self.comet_pre_classifier = torch.nn.Linear(1024, 1024)
        self.comet_dropout = torch.nn.Dropout(0.1)
        self.comet_classifier = torch.nn.Linear(1024, 1)

        # bert-score
        self.bert_score_pre_classifier = torch.nn.Linear(1024, 1024)
        self.bert_score_dropout = torch.nn.Dropout(0.1)
        self.bert_score_classifier = torch.nn.Linear(1024, 1)

        # HTER
        self.hter_pre_classifier = torch.nn.Linear(1024, 1024)
        self.hter_dropout = torch.nn.Dropout(0.1)
        self.hter_classifier = torch.nn.Linear(1024, 1)

        self.training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "chrf_eval_loss": [],
            "chrf_pearson": [],
            "chrf_spearman": [],
            "da_eval_loss": [],
            "da_pearson": [],
            "da_spearman": [],
            "comet_eval_loss": [],
            "comet_pearson": [],
            "comet_spearman": [],
            "bert_score_eval_loss": [],
            "bert_score_pearson": [],
            "bert_score_spearman": [],
            "hter_eval_loss": [],
            "hter_pearson": [],
            "hter_spearman": []
        }

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            label_type=None,
            comet=None,
            features=None,
            bert_score=None,
            hter=None

    ):

        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask)

        roberta_output = roberta_output[0][:, 0]

        chrf = self.chrf_pre_classifier(roberta_output)
        chrf = self.chrf_dropout(chrf)
        chrf = self.chrf_classifier(chrf)

        da = self.da_pre_classifier(roberta_output)
        da = self.da_dropout(da)
        da = self.da_classifier(da)

        comet = self.comet_pre_classifier(roberta_output)
        comet = self.comet_dropout(comet)
        comet = self.comet_classifier(comet)

        bert_score = self.bert_score_pre_classifier(roberta_output)
        bert_score = self.bert_score_dropout(bert_score)
        bert_score = self.bert_score_classifier(bert_score)

        hter = self.hter_pre_classifier(roberta_output)
        hter = self.hter_dropout(hter)
        hter = self.hter_classifier(hter)

        return chrf, da, comet, bert_score, hter

    def get_examples(self, df, label_type=0):
        if label_type == 1:
            examples = [
                InputExample(i, text_a, label=label, label_type=label_type, comet=comet, bert_score=bertscore)
                for i, (text_a, label, comet, bertscore) in enumerate(
                    zip(df["text_a"].astype(str), df["labels"], df["comet"], df['bertScore'])
                )
            ]
        else:  # label_type == 2
            examples = [
                InputExample(i, text_a, label=label, label_type=2, hter=hter)
                for i, (text_a, label, hter) in enumerate(
                        zip(df["text_a"].astype(str), df["labels"], df["hter"])
                )
            ]
        return examples