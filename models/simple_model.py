import sys
# sys.path.append("/cs/labs/oabend/shachar.don/pre-translationQE/clean_model/TransQuest/")

import numpy as np
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import random
from shared_paths_local import CACHE_DIR
from trainer import InputExample


class SimpleModel(nn.Module):

    def __init__(self, seed, multilingual=False):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        super(SimpleModel, self).__init__()
        model_name = "roberta-large"
        if multilingual:
            model_name = "xlm-" + model_name
        self.config = RobertaConfig.from_pretrained(model_name, cache_dir=CACHE_DIR, num_labels=1)
        self.roberta_preQE = RobertaModel.from_pretrained(model_name, cache_dir=CACHE_DIR, config=self.config)
        self.pre_classifier = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)

        self.training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_loss": [],
            "pearson": [],
            "spearman": []
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
        preQE_output = self.roberta_preQE(
            input_ids=input_ids,
            attention_mask=attention_mask)
        preQE_output = preQE_output[0][:, 0]

        x = self.pre_classifier(preQE_output)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def get_examples(self, df, label_type=0):
        examples = [
            InputExample(i, text_a, label=label, label_type=label_type)
            for i, (text_a, label) in enumerate(
                zip(df["text_a"].astype(str), df["labels"])
            )
        ]
        return examples