import sys
# sys.path.append("/cs/labs/oabend/shachar.don/pre-translationQE/clean_model/TransQuest/")

import numpy as np
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaClassificationHead
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import random
from shared_paths import CACHE_DIR


class ChooseModelReg(nn.Module):

    def __init__(self, seed, num_systems, multilingual=False):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        super(ChooseModelReg, self).__init__()
        model_name = "roberta-large"
        if multilingual:
            model_name = "xlm-" + model_name

        self.num_systems = num_systems
        self.config = RobertaConfig.from_pretrained(model_name, cache_dir=CACHE_DIR, regression=True, num_labels=1)
        self.roberta = RobertaModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.heads = nn.ModuleList()
        for i in range(self.num_systems):
            head = RobertaClassificationHead(self.config)
            self.heads.append(head)

        self.training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }

        # print(self.summary())

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
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask)
        sequence_output = outputs[0]
        # preQE_output = preQE_output[0][:, 0]

        all_outputs = []
        for system_head in self.heads:
            # print("sequence_output:", sequence_output.is_cuda)
            # print("system_head:", system_head.is_cuda)
            output_for_system = system_head(sequence_output)
            all_outputs.append(output_for_system)

        # preQE_output = self.roberta_preQE(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask)
        # preQE_output = preQE_output[0][:, 0]
        #
        # # x = self.dropout(preQE_output)
        # x = self.pre_classifier(preQE_output)
        # # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.classifier(x)

        return all_outputs
