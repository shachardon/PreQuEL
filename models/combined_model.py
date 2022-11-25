import numpy as np
import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,

)
from models.simple_model import SimpleModel
import random
from shared_paths_local import CACHE_DIR

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

ROBERTA_PARSER = "/cs/labs/oabend/shachar.don/pre-translationQE/robertnlp-enhanced-ud-parser/best_roberta_model_saved/"


class CombinedModel(SimpleModel):

    def __init__(self, seed, no_parser=False, multilingual=False):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        super(CombinedModel, self).__init__(seed)
        model_name = "roberta-large"
        if multilingual:
            model_name = "xlm-" + model_name
        self.config = RobertaConfig.from_pretrained(model_name, cache_dir=CACHE_DIR, num_labels=1)

        if no_parser:
            self.roberta_parser = RobertaModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        else:
            self.roberta_parser = RobertaModel.from_pretrained(ROBERTA_PARSER, cache_dir=CACHE_DIR)
        self.roberta_preQE = RobertaModel.from_pretrained(model_name, cache_dir=CACHE_DIR, config=self.config)


        self.pre_classifier = torch.nn.Linear(1024 * 2, 1024 * 2)
        self.dropout = torch.nn.Dropout(0.1)

        self.classifier = torch.nn.Linear(1024 * 2, 1)

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
        parser_output = self.roberta_parser(
            input_ids=input_ids,
            attention_mask=attention_mask)
        parser_output = parser_output[0][:, 0]

        preQE_output = self.roberta_preQE(
            input_ids=input_ids,
            attention_mask=attention_mask)
        preQE_output = preQE_output[0][:, 0]

        concat_output = torch.cat((preQE_output, parser_output), 1)

        x = self.pre_classifier(concat_output)
        x = self.dropout(x)
        x = self.classifier(x)

        return x