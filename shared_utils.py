from torch import cuda
import torch
import math
from sklearn.metrics import mean_absolute_error
from transformers import get_linear_schedule_with_warmup
from sklearn import preprocessing
from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse
import os

device = 'cuda' if cuda.is_available() else 'cpu'

# seed dict to reproduce experiments.
SEED_DICT = {0: 777, 1: 71, 2: 1129, 3: 3517, 4: 5849, 5: 7507, 6: 2749, 7: 1783, 8: 5443, 9: 7151,
             10: 6029, 11: 4139, 12: 5323, 13: 1997, 14: 2221}


def _get_inputs_dict(batch):
    if isinstance(batch[0], dict):
        inputs = {key: value.squeeze().to(device) for key, value in batch[0].items()}
        inputs["labels"] = batch[1].to(device)
    else:
        batch = tuple(t.to(device) for t in batch)

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "label_type": batch[4],
                  "comet": batch[5], "features": batch[6], "bert_score": batch[7], "hter": batch[8]}

        # XLM, DistilBERT and RoBERTa don't use segment_ids
        inputs["token_type_ids"] = (
            None
        )

    return inputs


def print_stat_sec(preds, labels):
    pearson = pearson_corr(preds, labels)
    spearman = spearman_corr(preds, labels)
    rmse_value = rmse(preds, labels)
    mae = mean_absolute_error(preds, labels)

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (
        rmse_value, mae, pearson, spearman)
    print(textstr)

    return pearson, spearman


def init_optimizer_and_scheduler(model, total=1000, epochs=3, lr=1e-5, adam_eps=1e-8, wramup_ratio=0.1):
    optimizer_grouped_parameters = []
    custom_parameter_names = set()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
    print(optimizer)

    t_total = total * epochs
    warmup_steps = math.ceil(t_total * wramup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    print(scheduler)
    return optimizer, scheduler


def print_stat(data_frame, real_column, prediction_column):
    data_frame = data_frame.sort_values(real_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (
        rmse_value, mae, pearson, spearman)

    print(textstr)


def fit(df, label):
    x = df[[label]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[label] = x_scaled
    return df


def un_fit(df, label):
    x = df[[label]].values.astype(float)
    zero_handred_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    x_unscaled = zero_handred_scaler.fit_transform(x)
    df[label] = x_unscaled
    return df


def fit_da(df, label):
    x = df[[label]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[label] = x_scaled
    return df


def un_fit_da(df, label):
    x = df[[label]].values.astype(float)
    zero_handred_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    x_unscaled = zero_handred_scaler.fit_transform(x)
    df[label] = x_unscaled
    return df


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


def rmse(preds, labels):
    return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no_parser', default=False, action='store_true')
    parser.add_argument('--extra_finetune', type=int, default=0)
    parser.add_argument('--da_weight', type=int)
    parser.add_argument('--gradient', type=int)
    parser.add_argument('--no_comet', default=False, action='store_true')
    parser.add_argument('--no_chrf', default=False, action='store_true')
    parser.add_argument('--model_type', type=str, default="simple")
    parser.add_argument('--data', type=str, nargs='+')
    parser.add_argument('--ref', default=False, action='store_true')
    parser.add_argument('--train_facebook', default=False, action='store_true')
    parser.add_argument('--dev_facebook', default=False, action='store_true')
    parser.add_argument('--test_only', type=str)
    parser.add_argument('--lang', type=str, default="de")
    parser.add_argument('--multilingual', default=False, action='store_true')
    parser.add_argument('--use_hter', default=False, action='store_true')
    parser.add_argument('--replace_hter', default=False, action='store_true')
    parser.add_argument('--replace_comet', default=False, action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--wmt19', default=False, action='store_true')
    parser.add_argument('--score_types', type=str, nargs='+')
    parser.add_argument('--systems', type=str, nargs='+')
    parser.add_argument('--search_seed', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    return args


def clean_progress_scores(progress_scores):
    for key in progress_scores:
        progress_scores[key] = []
    return progress_scores


def load_saved_model_if_exists(model, output_dir, fold, args, extra_finetune=False):
    # check if we have a saved model.

    epochs_to_run = args.extra_finetune if extra_finetune else args.epochs
    trained_epochs = 0
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        path = "%s/best_model_%d.bin" % (output_dir, fold)
        if extra_finetune:
            path = "%s/best_model_finetune_%d.bin" % (output_dir, fold)
        if os.path.exists(path):
            print("loading model from existing checkpoint:", path)
            model.load_state_dict(torch.load(path))
            trained_epochs = epochs_to_run
        else:
            for j in range(epochs_to_run + 1, 0, -1):
                path = "%s/pytorch_model_%d_%d.bin" % (output_dir, j, fold)
                if extra_finetune:
                    path = "%s/pytorch_model_finetune_%d_%d.bin" % (output_dir, j, fold)
                if os.path.exists(path):
                    print("loading model from existing checkpoint:", path)
                    model.load_state_dict(torch.load(path))
                    trained_epochs = j
                    break
    else:
        os.makedirs(output_dir, exist_ok=True)
    return model, trained_epochs
