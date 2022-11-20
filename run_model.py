#!/usr/bin/python3
import os
os.environ['TRANSFORMERS_CACHE'] = '/cs/labs/oabend/shachar.don/pre-translationQE/hg_cache'
import sys
sys.path.append("/cs/labs/oabend/shachar.don/pre-translationQE/my_model/")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import gc
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from shared_utils import _get_inputs_dict, print_stat_sec, \
    init_optimizer_and_scheduler, print_stat, fit, un_fit, fit_da, un_fit_da, parse_train_args, clean_progress_scores, \
    SEED_DICT, load_saved_model_if_exists
from shared_paths import TRAIN_WMT, DEV_WMT, TRAIN_TATOEBA, DEV_TATOEBA, TRAIN_NEWS20, DEV_NEWS20, \
    TRAIN_BIBLE, DEV_BIBLE, TRAIN_GLOBAL_VOICES, DEV_GLOBAL_VOICES, FACEBOOK_TRAIN_WMT, \
    FACEBOOK_DEV_WMT, FACEBOOK_TRAIN_TATOEBA, FACEBOOK_DEV_TATOEBA, FACEBOOK_TRAIN_NEWS20, FACEBOOK_DEV_NEWS20, \
    FACEBOOK_TRAIN_BIBLE, FACEBOOK_DEV_BIBLE, FACEBOOK_TRAIN_GLOBAL_VOICES, FACEBOOK_DEV_GLOBAL_VOICES, \
    TRAIN_DA, DEV_DA, ZH_TRAIN_WMT, ZH_TRAIN_NEWS20, ZH_DEV_NEWS20, ZH_TRAIN_BIBLE, ZH_DEV_WMT, \
    ZH_DEV_BIBLE, ZH_DA_TRAIN, ZH_DA_DEV, RE_TRAIN_WMT, RE_DEV_WMT, RE_TRAIN_TATOEBA, RE_DEV_TATOEBA, RE_TRAIN_NEWS20, \
    RE_DEV_NEWS20, RE_TRAIN_BIBLE, RE_DEV_BIBLE, RE_TRAIN_GLOBAL, RE_DEV_GLOBAL, ET_ET_TRAIN_DA, ET_ET_DEV_DA, \
    WMT19_TRAIN, WMT19_DEV
from shared_data_utils import prepare_data
from data_reader import read_annotated_file, read_test_file
from trainer import Trainer
from models.simple_model import SimpleModel
from models.combined_model import CombinedModel
from models.multiple_model import MultipleModel

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Set pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

SEED = 776
RESULT_FILE = "result.tsv"
SUBMISSION_FILE = "predictions.txt"
RESULT_IMAGE = "result.jpg"
MODEL_NAME = "roberta-large"

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-5
ADAM_EPSILON = 1e-8
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEP = 1
COMET = 1
CHRF = 1
HTER = 0
BERT_SCORE = 1
DA_WEIGHT = 1
seed = SEED

dev_name_dict = {
    FACEBOOK_DEV_WMT: "facebook_WMT",
    FACEBOOK_DEV_BIBLE: "facebook_bible",
    FACEBOOK_DEV_TATOEBA: "facebook_tatoeba",
    FACEBOOK_DEV_NEWS20: "facebook_news20",
    FACEBOOK_DEV_GLOBAL_VOICES: "facebook_global_voices",
    DEV_WMT: "wmt",
    DEV_BIBLE: "bible",
    DEV_TATOEBA: "tatoeba",
    DEV_NEWS20: "news20",
    DEV_GLOBAL_VOICES: "global_voices",
    ZH_DEV_WMT: "zh_wmt",
    ZH_DEV_NEWS20: "zh_news20",
    ZH_DEV_BIBLE: "zh_bible",
    RE_DEV_WMT: "re_wmt",
    RE_DEV_BIBLE: "re_bible",
    RE_DEV_TATOEBA: "re_tatoeba",
    RE_DEV_NEWS20: "re_news20",
    RE_DEV_GLOBAL: "re_global_voices",
}


def get_model(model_type, is_multilingual=False, seed_to_use=SEED, extra_args=None):
    if model_type == "simple":
        return SimpleModel(seed_to_use, multilingual=is_multilingual)
    if model_type == "combined":
        return CombinedModel(seed_to_use, multilingual=is_multilingual, no_parser=extra_args.no_parser)
    if model_type == "multiple":
        return MultipleModel(seed_to_use, multilingual=is_multilingual)


def run_epoch(model, optimizer, scheduler, training_loader, cur_epoch, all_epochs=10, eval_dataloader=None):
    loss_function = torch.nn.MSELoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    steps_in_epoch = len(training_loader)
    if steps_in_epoch > 10000:
        run_val = 3000
    else:
        run_val = 300

    best_loss = -1
    counter = 0
    last_epoch = False

    model.train()
    for _, data in tqdm(enumerate(training_loader, 0), desc=f"Running Epoch {cur_epoch} of {all_epochs}"):
        inputs = _get_inputs_dict(data)
        # print("inputs", inputs)
        outputs = model(**inputs)
        targets = inputs['labels'].to(device, dtype=torch.float)

        # print("train - targets:", targets, "outputs:", outputs)
        loss = loss_function(outputs.view(-1), targets.view(-1))
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if (_ % run_val) == 0:
            loss_step = tr_loss/nb_tr_steps
            print("Training Loss per", run_val, "steps:",  loss_step)
            sys.stdout.flush()

            model.training_progress_scores["global_step"].append((steps_in_epoch * (cur_epoch - 1)) + nb_tr_steps)
            model.training_progress_scores["train_loss"].append(loss_step)

            if eval_dataloader:
                print("\neval")
                eval_loss, pearson, spearman, outputs, targets = evaluate(model, eval_dataloader, cur_epoch)
                model.training_progress_scores["eval_loss"].append(eval_loss)
                model.training_progress_scores["pearson"].append(pearson)
                model.training_progress_scores["spearman"].append(spearman)
                
                if best_loss < 0 or best_loss > eval_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), "%s/best_model.bin" % output_dir)
                    counter = 0
                else:
                    counter += 1
                    print("model didn't improve for", counter, "eval steps")
                    if counter > 9:
                        last_epoch = True
                        print("exiting training")
                        break

            print(model.training_progress_scores)

        loss.backward()
        # When using GPU
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

    epoch_loss = tr_loss/nb_tr_steps
    writer.add_scalar("Loss_/train", epoch_loss, cur_epoch)
    print(f"Training Loss Epoch: {epoch_loss}")

    model.training_progress_scores["global_step"].append((steps_in_epoch * (cur_epoch - 1)) + nb_tr_steps)
    model.training_progress_scores["train_loss"].append(epoch_loss)

    if eval_dataloader:
        print("\neval")
        eval_loss, pearson, spearman, outputs, targets = evaluate(model, eval_dataloader, cur_epoch)
        model.training_progress_scores["eval_loss"].append(eval_loss)
        model.training_progress_scores["pearson"].append(pearson)
        model.training_progress_scores["spearman"].append(spearman)
    
    model.load_state_dict(torch.load("%s/best_model.bin" % output_dir))  # we want to finish the epoch with the best model loaded.
    return last_epoch


def run_multitask_epoch(model, optimizer, scheduler, training_loader, cur_epoch, all_epochs=10,
                        eval_dataloader_chrf=None, eval_dataloader_da=None):
    loss_function = torch.nn.MSELoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    steps_in_epoch = len(training_loader)
    print("steps_in_epoch:", steps_in_epoch)
    if steps_in_epoch > 10000:
        run_val = 3000
    else:
        run_val = 300
    print("run_val:", run_val)

    best_loss = -1
    counter = 0
    last_epoch = False

    for _, data in tqdm(enumerate(training_loader, 0), desc=f"Running Epoch {cur_epoch} of {all_epochs}"):
        inputs = _get_inputs_dict(data)
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5 = model(**inputs)
        targets = inputs['labels'].to(device, dtype=torch.float)
        comets = inputs['comet'].to(device, dtype=torch.float)
        bert_scores = inputs['bert_score'].to(device, dtype=torch.float)
        labels_type = inputs['label_type']

        loss = 0

        outputs_1_flat = outputs_1.view(-1)  # chrf
        outputs_2_flat = outputs_2.view(-1)  # da
        outputs_3_flat = outputs_3.view(-1)  # comet
        outputs_4_flat = outputs_4.view(-1)  # bert score
        outputs_5_flat = outputs_5.view(-1)  # hter

        chrf_mask = labels_type == 1
        da_mask = labels_type == 2

        if torch.any(chrf_mask):
            loss += (loss_function(outputs_1_flat[chrf_mask], targets[chrf_mask]) * CHRF)  # CHRf
            loss += (loss_function(outputs_3_flat[chrf_mask], comets[chrf_mask]) * COMET)  # COMET
            loss += (loss_function(outputs_4_flat[chrf_mask], bert_scores[chrf_mask]) * BERT_SCORE)  # bert score

        if torch.any(da_mask):
            loss += (loss_function(outputs_2_flat[da_mask], targets[da_mask]) * DA_WEIGHT)  # DA
            if args.use_hter:
                loss += (loss_function(outputs_5_flat[da_mask], targets[da_mask]) * DA_WEIGHT)  # HTER

        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if (nb_tr_steps % run_val) == 0:
            print("nb_tr_steps:", nb_tr_steps)
            loss_step = tr_loss/nb_tr_steps  # Shachar: when training on comet and chrf then the number of examples is actually double. should we take care of that?
            print("Training Loss per", run_val, "steps:", loss_step)
            sys.stdout.flush()

            model.training_progress_scores["global_step"].append((steps_in_epoch * (cur_epoch - 1)) + nb_tr_steps)
            model.training_progress_scores["train_loss"].append(loss_step)

            eval_loss = 0

            if eval_dataloader_da:
                print("\neval DA")
                da_results, hter_results = evaluate_multitask(model, eval_dataloader_da, cur_epoch, head_type=2)
                epoch_loss, pearson, spearman, outputs, targets = da_results
                model.training_progress_scores["da_eval_loss"].append(epoch_loss)
                model.training_progress_scores["da_pearson"].append(pearson)
                model.training_progress_scores["da_spearman"].append(spearman)

                epoch_loss, pearson, spearman, outputs, targets = hter_results
                model.training_progress_scores["hter_eval_loss"].append(epoch_loss)
                model.training_progress_scores["hter_pearson"].append(pearson)
                model.training_progress_scores["hter_spearman"].append(spearman)

                if not eval_dataloader_chrf:
                    eval_loss += da_results[0]
                    if args.use_hter:
                        eval_loss += hter_results[0]
                    if best_loss < 0 or best_loss > eval_loss:
                        best_loss = eval_loss
                        torch.save(model.state_dict(), "%s/best_model.bin" % output_dir)
                        counter = 0
                    else:
                        counter += 1
                        print("model didn't improve for", counter, "eval steps")
                        if counter > 9:
                            last_epoch = True
                            print("exiting training")
                            break

            if eval_dataloader_chrf:
                print("\neval CHRF + COMET")
                chrf_results, comet_results, bert_score_results = evaluate_multitask(model, eval_dataloader_chrf, cur_epoch, head_type=1)

                model.training_progress_scores["chrf_eval_loss"].append(chrf_results[0])
                model.training_progress_scores["chrf_pearson"].append(chrf_results[1])
                model.training_progress_scores["chrf_spearman"].append(chrf_results[2])

                model.training_progress_scores["comet_eval_loss"].append(comet_results[0])
                model.training_progress_scores["comet_pearson"].append(comet_results[1])
                model.training_progress_scores["comet_spearman"].append(comet_results[2])

                model.training_progress_scores["bert_score_eval_loss"].append(bert_score_results[0])
                model.training_progress_scores["bert_score_pearson"].append(bert_score_results[1])
                model.training_progress_scores["bert_score_spearman"].append(bert_score_results[2])

                eval_loss += (chrf_results[0] + comet_results[0] + bert_score_results[0])
                if best_loss < 0 or best_loss > eval_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), "%s/best_model.bin" % output_dir)
                    counter = 0
                else:
                    counter += 1
                    print("model didn't improve for", counter, "eval steps")
                    if counter > 9:
                        last_epoch = True
                        print("exiting training")
                        break

            print(model.training_progress_scores)

        loss.backward()
        if (nb_tr_steps % GRADIENT_ACCUMULATION_STEP) == 0:
            # # When using GPU
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

    epoch_loss = tr_loss/nb_tr_steps
    writer.add_scalar("Loss_/train", epoch_loss, cur_epoch)
    print(f"Training Loss Epoch: {epoch_loss}")

    model.training_progress_scores["global_step"].append((steps_in_epoch * (cur_epoch - 1)) + nb_tr_steps)
    model.training_progress_scores["train_loss"].append(epoch_loss)

    if eval_dataloader_chrf:
        print("\neval CHRF + COMET")
        chrf_results, comet_results, bert_score_results = evaluate_multitask(model, eval_dataloader_chrf, cur_epoch, head_type=1)
        model.training_progress_scores["chrf_eval_loss"].append(chrf_results[0])
        model.training_progress_scores["chrf_pearson"].append(chrf_results[1])
        model.training_progress_scores["chrf_spearman"].append(chrf_results[2])

        model.training_progress_scores["comet_eval_loss"].append(comet_results[0])
        model.training_progress_scores["comet_pearson"].append(comet_results[1])
        model.training_progress_scores["comet_spearman"].append(comet_results[2])

        model.training_progress_scores["bert_score_eval_loss"].append(bert_score_results[0])
        model.training_progress_scores["bert_score_pearson"].append(bert_score_results[1])
        model.training_progress_scores["bert_score_spearman"].append(bert_score_results[2])

    if eval_dataloader_da:
        print("\neval DA")
        da_results, hter_results = evaluate_multitask(model, eval_dataloader_da, cur_epoch, head_type=2)
        epoch_loss, pearson, spearman, outputs, targets = da_results
        model.training_progress_scores["da_eval_loss"].append(epoch_loss)
        model.training_progress_scores["da_pearson"].append(pearson)
        model.training_progress_scores["da_spearman"].append(spearman)

        epoch_loss, pearson, spearman, outputs, targets = hter_results
        model.training_progress_scores["hter_eval_loss"].append(epoch_loss)
        model.training_progress_scores["hter_pearson"].append(pearson)
        model.training_progress_scores["hter_spearman"].append(spearman)

    model.load_state_dict(torch.load("%s/best_model.bin" % output_dir))  # we want to finish the epoch with the best model loaded.
    return last_epoch


def evaluate(model, testing_loader, epoch):
    outputs_all = []
    target_all = []

    loss_function = torch.nn.MSELoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.eval()
    for _, data in tqdm(enumerate(testing_loader, 0), desc=f"Running Validation"):
        with torch.no_grad():
            inputs = _get_inputs_dict(data)
            outputs = model(**inputs)
            targets = inputs['labels'].to(device, dtype=torch.float)

            target_all += targets.detach().cpu().numpy().flatten().tolist()
            outputs_all += outputs.detach().cpu().numpy().flatten().tolist()

            loss = loss_function(outputs.view(-1), targets.view(-1))
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if (_ % 200) == 0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Validation Loss per 200 steps: {loss_step}")
                sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()

    model.train()
    epoch_loss = tr_loss/nb_tr_steps
    writer.add_scalar("Loss_/valid", epoch_loss, epoch)
    print(f"Validation Loss Epoch: {epoch_loss}")
    pearson, spearman = print_stat_sec(np.array(outputs_all), np.array(target_all))
    return epoch_loss, pearson, spearman, np.array(outputs_all), np.array(target_all)


def evaluate_multitask(model, testing_loader, epoch, head_type):
    outputs_all = []
    target_all = []
    comet_outputs_all = []
    comet_target_all = []
    bert_score_outputs_all = []
    bert_score_target_all = []
    hter_outputs_all = []
    hter_target_all = []

    loss_function = torch.nn.MSELoss()
    tr_loss = 0
    tr_loss_comet = 0
    tr_loss_bert_score = 0
    tr_loss_hter = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.eval()
    for _, data in tqdm(enumerate(testing_loader, 0), desc=f"Running Validation"):
        with torch.no_grad():
            inputs = _get_inputs_dict(data)
            all_outputs = model(**inputs)

            if head_type == 1:  # not just CHRF, we add also COMET! (and bertScore!!)
                comet_targets = inputs['comet'].to(device, dtype=torch.float)
                comet_outputs = all_outputs[2]
                comet_outputs_all += comet_outputs.detach().cpu().numpy().flatten().tolist()
                comet_target_all += comet_targets.detach().cpu().numpy().flatten().tolist()
                loss_comet = loss_function(comet_targets.view(-1), comet_outputs.view(-1))
                tr_loss_comet += loss_comet.item()

                bert_score_targets = inputs['bert_score'].to(device, dtype=torch.float)
                bert_score_outputs = all_outputs[3]
                bert_score_outputs_all += bert_score_outputs.detach().cpu().numpy().flatten().tolist()
                bert_score_target_all += bert_score_targets.detach().cpu().numpy().flatten().tolist()
                loss_bert_score = loss_function(bert_score_targets.view(-1), bert_score_outputs.view(-1))
                tr_loss_bert_score += loss_bert_score.item()

            if head_type == 2:  # we add also HTER
                hter_targets = inputs['hter'].to(device, dtype=torch.float)
                hter_outputs = all_outputs[4]
                hter_outputs_all += hter_outputs.detach().cpu().numpy().flatten().tolist()
                hter_target_all += hter_targets.detach().cpu().numpy().flatten().tolist()
                loss_hter = loss_function(hter_targets.view(-1), hter_outputs.view(-1))
                tr_loss_hter += loss_hter.item()

            targets = inputs['labels'].to(device, dtype=torch.float)
            outputs = all_outputs[head_type - 1]
            target_all += targets.detach().cpu().numpy().flatten().tolist()
            outputs_all += outputs.detach().cpu().numpy().flatten().tolist()
            loss = loss_function(outputs.view(-1), targets.view(-1))
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if (nb_tr_steps % 200) == 0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Validation Loss per 200 steps: {loss_step}")
                sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()

    epoch_loss = tr_loss/nb_tr_steps
    writer.add_scalar("Loss_/valid", epoch_loss, epoch)
    print(f"Validation Loss Epoch: {epoch_loss}")
    pearson, spearman = print_stat_sec(np.array(outputs_all), np.array(target_all))
    model.train()

    if head_type == 1:
        comet_epoch_loss = tr_loss_comet/nb_tr_steps
        comet_pearson, comet_spearman = print_stat_sec(np.array(comet_outputs_all), np.array(comet_target_all))
        bert_score_epoch_loss = tr_loss_bert_score/nb_tr_steps
        bert_score_pearson, bert_score_spearman = print_stat_sec(np.array(bert_score_outputs_all), np.array(bert_score_target_all))

        return [(epoch_loss, pearson, spearman, np.array(outputs_all), np.array(target_all)),
                (comet_epoch_loss, comet_pearson, comet_spearman, np.array(comet_outputs_all), np.array(comet_target_all)),
                (bert_score_epoch_loss, bert_score_pearson, bert_score_spearman, np.array(bert_score_outputs_all), np.array(bert_score_target_all))]

    hter_epoch_loss = tr_loss_hter/nb_tr_steps
    hter_pearson, hter_spearman = print_stat_sec(np.array(hter_outputs_all), np.array(hter_target_all))

    return [(epoch_loss, pearson, spearman, np.array(outputs_all), np.array(target_all)),
            (hter_epoch_loss, hter_pearson, hter_spearman, np.array(hter_outputs_all), np.array(hter_target_all))]


def predict(model, testing_loader):
    outputs_all = []
    target_all = []

    model.eval()
    for _, data in tqdm(enumerate(testing_loader, 0), desc=f"Running Validation"):
        with torch.no_grad():
            inputs = _get_inputs_dict(data)
            outputs = model(**inputs)
            targets = inputs['labels'].to(device, dtype=torch.float)

            target_all += targets.detach().cpu().numpy().flatten().tolist()
            outputs_all += outputs.detach().cpu().numpy().flatten().tolist()

        torch.cuda.empty_cache()

    return np.array(outputs_all), np.array(target_all)


def predict_multitask(model, testing_loader, head_type):
    outputs_all = []
    target_all = []
    comet_outputs_all = []
    comet_target_all = []
    bert_score_outputs_all = []
    bert_score_target_all = []
    hter_outputs_all = []
    hter_target_all = []

    model.eval()
    for _, data in tqdm(enumerate(testing_loader, 0), desc=f"Running Validation"):
        with torch.no_grad():
            inputs = _get_inputs_dict(data)
            all_outputs = model(**inputs)

            if head_type == 1:  # not just CHRF, we add also COMET! (and bertScore!!)
                comet_targets = inputs['comet'].to(device, dtype=torch.float)
                comet_outputs = all_outputs[2]
                comet_outputs_all += comet_outputs.detach().cpu().numpy().flatten().tolist()
                comet_target_all += comet_targets.detach().cpu().numpy().flatten().tolist()

                bert_score_targets = inputs['bert_score'].to(device, dtype=torch.float)
                bert_score_outputs = all_outputs[3]
                bert_score_outputs_all += bert_score_outputs.detach().cpu().numpy().flatten().tolist()
                bert_score_target_all += bert_score_targets.detach().cpu().numpy().flatten().tolist()


            if head_type == 2:  # we add also HTER
                hter_targets = inputs['hter'].to(device, dtype=torch.float)
                hter_outputs = all_outputs[4]
                hter_outputs_all += hter_outputs.detach().cpu().numpy().flatten().tolist()
                hter_target_all += hter_targets.detach().cpu().numpy().flatten().tolist()


            targets = inputs['labels'].to(device, dtype=torch.float)
            outputs = all_outputs[head_type - 1]
            target_all += targets.detach().cpu().numpy().flatten().tolist()
            outputs_all += outputs.detach().cpu().numpy().flatten().tolist()

        gc.collect()
        torch.cuda.empty_cache()

    model.train()

    if head_type == 1:

        return [(np.array(outputs_all), np.array(target_all)),
                (np.array(comet_outputs_all), np.array(comet_target_all)),
                (np.array(bert_score_outputs_all), np.array(bert_score_target_all))]

    return [(np.array(outputs_all), np.array(target_all)),
            (np.array(hter_outputs_all), np.array(hter_target_all))]


def test_only(input_path, ref, output_dir, args, epoch=0, label_type=1):
    # load df
    paths = []
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            filename = os.fsdecode(file)
            print(filename)
            path = os.path.join(input_path, filename)
            paths.append(path)
    else:
        paths.append(input_path)

    for file_path in paths:
        if file_path.endswith(".txt"):
            data = [line.strip() for line in open(file_path, 'r')]
            print(data)
            df = pd.DataFrame(
                {'index': np.zeros(len(data)),
                 'original': data,
                 'z_mean': np.zeros(len(data)),
                 'comet': np.zeros(len(data)),
                 'bertScore': np.zeros(len(data)),
                 'unigram': np.zeros(len(data)),
                 'bigram': np.zeros(len(data)),
                 '3gram': np.zeros(len(data)),
                 '4gram': np.zeros(len(data)),
                 'lan': np.zeros(len(data)),
                 'hter': np.zeros(len(data)),
                 })
        elif "test20" in file_path:
            df = read_test_file(file_path)
            df['z_mean'] = 0
            df['comet'] = 0
            df['bertScore'] = 0
            df['hter'] = 0
        else:
            print(file_path)
            df = read_annotated_file(file_path, features=(True if args.model_type == "features" else False), ref=ref,
                                     comet=(True if label_type == 1 else False),
                                     bertScore=(True if label_type == 1 else False))

        df = df.rename(columns={'original': 'text_a', 'z_mean': 'labels'}).dropna()
        df = df.reset_index(drop=True)
        df = fit(df, 'labels')

        print(df)

        # init preds table
        preds = np.zeros((4, len(df), args.folds))
        print(preds.shape)

        # load model for fold
        for fold in range(1, args.folds + 1):

            multilingual = (args.ref if args.lang != "en" else (not args.ref)) or args.multilingual
            model = get_model(args.model_type, is_multilingual=multilingual, seed_to_use=seed, extra_args=args)
            if epoch > 0:
                model_path = "%s/pytorch_model_%d_%d.bin" % (output_dir, epoch, fold)
            else:
                model_path = "%s/best_model_%d.bin" % (output_dir, fold)
            print("path:", model_path)

            if os.path.exists(model_path):
                print("loading model from existing checkpoint:", model_path)
                model.load_state_dict(torch.load(model_path))
            else:
                print("Error: didn't find the model!")
            model.to(device)

            model_name = "roberta-large"
            if multilingual:
                model_name = "xlm-" + model_name

            # prepare data
            trainer = Trainer('roberta', model_name, num_labels=1)
            examples = model.get_examples(df, label_type=label_type)
            dataset = trainer.load_and_cache_examples(
                examples, evaluate=True, verbose=False, silent=True
            )
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=VALID_BATCH_SIZE)

            # predict
            if label_type == 1:  # CHRF
                if args.model_type == "multiple":
                    chrf_results, comet_results, bert_score_results = predict_multitask(model, dataloader, head_type=1)
                    outputs, targets = chrf_results
                    preds[0, :len(df), fold - 1] = outputs
                    outputs, targets = comet_results
                    preds[1, :len(df), fold - 1] = outputs
                    outputs, targets = bert_score_results
                    preds[2, :len(df), fold - 1] = outputs
                else:
                    outputs, targets = predict(model, dataloader)
                    preds[0, :len(df), fold - 1] = outputs

            if label_type == 2:  # DA
                if args.model_type == "multiple":
                    results_da, results_hter = predict_multitask(model, dataloader, head_type=2)
                    outputs, targets = results_da
                    preds[0, :len(df), fold - 1] = outputs
                    outputs, targets = results_hter
                    preds[1, :len(df), fold - 1] = outputs

                else:
                    outputs, targets = predict(model, dataloader)
                    preds[0, :len(df), fold - 1] = outputs

            if args.extra_finetune and label_type == 2:
                model_path = "%s/best_model_finetune_%d.bin" % (output_dir, fold)
                if os.path.exists(model_path):
                    print("loading model from existing checkpoint:", model_path)
                    model.load_state_dict(torch.load(model_path))
                else:
                    print("Error: didn't find the model!")

                if args.model_type == "multiple":
                    results_da, results_hter = predict_multitask(model, dataloader, head_type=2)
                    outputs, targets = results_da
                    preds[2, :len(df), fold - 1] = outputs
                    outputs, targets = results_hter
                    preds[3, :len(df), fold - 1] = outputs
                else:
                    outputs, targets = predict(model, dataloader)
                    preds[1, :len(df), fold - 1] = outputs

        # save results
        file_name = os.path.basename(file_path)
        file_name = file_name[:-4] if file_name.endswith(".txt") else file_name
        print(file_name)
        output_test_dir = output_dir + "/test_%s/" % file_name
        os.makedirs(output_test_dir, exist_ok=True)

        if args.model_type != "multiple":
            print("preds", preds.mean())
            out_df = preds[0].mean(axis=1)
            out_df = pd.DataFrame({'predictions': out_df, 'labels': df['labels']})
            # out_df = un_fit(out_df, 'predictions')
            # out_df = un_fit(out_df, 'labels')
            out_df.to_csv(os.path.join(output_test_dir, "chrf" if label_type == 1 else "da"), header=True, sep='\t', index=False, encoding='utf-8')
        else:
            scores = ["labels", "comet", "bertScore"] if label_type == 1 else ["labels", "hter"]
            for i, score in enumerate(scores):
                out_df = preds[i].mean(axis=1)
                out_df = pd.DataFrame({'predictions': out_df, 'labels': df[score]})
                # out_df = un_fit(out_df, 'predictions')
                # out_df = un_fit(out_df, 'labels')
                out_df.to_csv(os.path.join(output_test_dir, score), header=True, sep='\t', index=False, encoding='utf-8')

        if args.extra_finetune and label_type == 2:
            if args.model_type != "multiple":
                out_df = preds[1].mean(axis=1)
                out_df = pd.DataFrame({'predictions': out_df, 'labels': df['labels']})
                # out_df = un_fit(out_df, 'predictions')
                # out_df = un_fit(out_df, 'labels')
                out_df.to_csv(os.path.join(output_test_dir, "da_finetune"), header=True, sep='\t', index=False, encoding='utf-8')
            else:
                scores = ["labels", "hter"]
                for i, score in enumerate(scores):
                    out_df = preds[2 + i].mean(axis=1)
                    out_df = pd.DataFrame({'predictions': out_df, 'labels': df[score]})
                    # out_df = un_fit(out_df, 'predictions')
                    # out_df = un_fit(out_df, 'labels')
                    out_df.to_csv(os.path.join(output_test_dir, score + "_finetune"), header=True, sep='\t', index=False, encoding='utf-8')


def get_train_paths(args):
    # train data
    if args.data:
        print("args.data:", args.data)

        if args.train_facebook:
            train_path_dict = {
                "wmt": FACEBOOK_TRAIN_WMT,
                "wmt2020": FACEBOOK_TRAIN_NEWS20,
                "tatoeba": FACEBOOK_TRAIN_TATOEBA,
                "global": FACEBOOK_TRAIN_GLOBAL_VOICES,
                "bible": FACEBOOK_TRAIN_BIBLE
            }
        else:
            if args.lang == "de":
                train_path_dict = {
                    "wmt": TRAIN_WMT,
                    "wmt2020": TRAIN_NEWS20,
                    "tatoeba": TRAIN_TATOEBA,
                    "global": TRAIN_GLOBAL_VOICES,
                    "bible": TRAIN_BIBLE
                }
            elif args.lang == "zh":
                train_path_dict = {
                    "wmt": ZH_TRAIN_WMT,
                    "wmt2020": ZH_TRAIN_NEWS20,
                    "bible": ZH_TRAIN_BIBLE
                }
            elif args.lang == "en":
                train_path_dict = {
                    "wmt": RE_TRAIN_WMT,
                    "wmt2020": RE_TRAIN_NEWS20,
                    "tatoeba": RE_TRAIN_TATOEBA,
                    "global": RE_TRAIN_GLOBAL,
                    "bible": RE_TRAIN_BIBLE
                }
            elif args.lang == "et":
                train_path_dict = {
                    "wmt": TRAIN_WMT,
                    "wmt2020": TRAIN_NEWS20,
                    "tatoeba": TRAIN_TATOEBA,
                    "global": TRAIN_GLOBAL_VOICES,
                    "bible": TRAIN_BIBLE
                }
        train_paths = [train_path_dict[name] for name in args.data]

    else:
        if args.train_facebook:
            train_paths = [FACEBOOK_TRAIN_WMT, FACEBOOK_TRAIN_NEWS20, FACEBOOK_TRAIN_TATOEBA,
                           FACEBOOK_TRAIN_GLOBAL_VOICES, FACEBOOK_TRAIN_BIBLE]
        else:
            if args.lang == "de":
                train_paths = [TRAIN_WMT, TRAIN_NEWS20, TRAIN_TATOEBA, TRAIN_GLOBAL_VOICES, TRAIN_BIBLE]
            elif args.lang == "zh":
                train_paths = [ZH_TRAIN_WMT, ZH_TRAIN_NEWS20, ZH_TRAIN_BIBLE]
            elif args.lang == "en":
                train_paths = [RE_TRAIN_WMT, RE_TRAIN_NEWS20, RE_TRAIN_TATOEBA, RE_TRAIN_GLOBAL, RE_TRAIN_BIBLE]
            elif args.lang == "et":  # we put de data here, but don't use it for training
                EPOCHS = 0
                train_paths = [TRAIN_WMT, TRAIN_NEWS20, TRAIN_TATOEBA, TRAIN_GLOBAL_VOICES, TRAIN_BIBLE]
    return train_paths


def get_train_data(args):
    train_paths = get_train_paths(args)
    train_datasets = [read_annotated_file(path, features=False, ref=args.ref,
                                          replace_comet=args.replace_comet) for path in train_paths]
    train_datasets = prepare_data(train_datasets)
    if args.limit:
        train_datasets = [dataset.sample(n=args.limit) for dataset in train_datasets]
    train = pd.concat(train_datasets, ignore_index=True)
    train = train.reset_index(drop=True)
    return train


def get_train_and_dev_da_data(args):
    if args.lang == "de":
        if args.wmt19:
            train_da = read_annotated_file(WMT19_TRAIN, features=False, ref=False, comet=False, bertScore=False,
                                           hter=True, replace_hter=args.replace_hter)
            dev_da = read_annotated_file(WMT19_DEV, features=False, ref=False, comet=False, bertScore=False,
                                         hter=True, replace_hter=args.replace_hter)
        else:
            train_da = read_annotated_file(TRAIN_DA, features=False, ref=False, comet=False, bertScore=False,
                                           hter=True, replace_hter=args.replace_hter)
            dev_da = read_annotated_file(DEV_DA, features=False, ref=False, comet=False, bertScore=False,
                                         hter=True, replace_hter=args.replace_hter)
    elif args.lang == "zh":
        train_da = read_annotated_file(ZH_DA_TRAIN, features=False, ref=False, comet=False,
                                       bertScore=False, hter=True, replace_hter=args.replace_hter)
        dev_da = read_annotated_file(ZH_DA_DEV, features=False, ref=False, comet=False,
                                     bertScore=False, hter=True, replace_hter=args.replace_hter)

    elif args.lang == "en":  # we put here the de data, but don't use it (no finetune)
        train_da = read_annotated_file(TRAIN_DA, features=False, ref=False, comet=False, bertScore=False,
                                       hter=True, replace_hter=args.replace_hter)
        dev_da = read_annotated_file(DEV_DA, features=False, ref=False, comet=False, bertScore=False,
                                     hter=True, replace_hter=args.replace_hter)
    elif args.lang == "et":
        train_da = read_annotated_file(ET_ET_TRAIN_DA, features=False, ref=False, comet=False,
                                       bertScore=False,
                                       hter=True, replace_hter=args.replace_hter)
        dev_da = read_annotated_file(ET_ET_DEV_DA, features=False, ref=False, comet=False, bertScore=False,
                                     hter=True, replace_hter=args.replace_hter)

    train_da, dev_da = prepare_data([train_da, dev_da], da=True)
    return train_da, dev_da


def get_dev_data(args):
    if args.dev_facebook:
        dev_paths = [FACEBOOK_DEV_WMT, FACEBOOK_DEV_TATOEBA, FACEBOOK_DEV_NEWS20, FACEBOOK_DEV_BIBLE,
                     FACEBOOK_DEV_GLOBAL_VOICES]
    else:
        if args.lang == "de":
            dev_paths = [DEV_WMT, DEV_NEWS20, DEV_BIBLE, DEV_TATOEBA, DEV_GLOBAL_VOICES]
        elif args.lang == "zh":
            dev_paths = [ZH_DEV_WMT, ZH_DEV_NEWS20, ZH_DEV_BIBLE]
        elif args.lang == "en":
            dev_paths = [RE_DEV_WMT, RE_DEV_NEWS20, RE_DEV_BIBLE, RE_DEV_TATOEBA, RE_DEV_GLOBAL]
        elif args.lang == "et":  # we put here the de data, but don't use it (only finetune on da)
            dev_paths = [DEV_WMT]

    dev_datasets = [read_annotated_file(path, features=False, ref=args.ref,
                                        replace_comet=args.replace_comet) for path in dev_paths]
    dev_datasets = prepare_data(dev_datasets)

    dev = pd.concat(dev_datasets, ignore_index=True)
    dev = dev.reset_index(drop=True)

    return dev_paths, dev_datasets, dev


def fit_data(train, dev, dev_datasets, train_da, dev_da):
    train = fit(train, 'labels')
    dev = fit(dev, 'labels')
    dev_datasets = [fit(dev_split, 'labels') for dev_split in dev_datasets]

    train = fit(train, 'comet')
    dev = fit(dev, 'comet')
    dev_datasets = [fit(dev_split, 'comet') for dev_split in dev_datasets]

    train = fit(train, 'bertScore')
    dev = fit(dev, 'bertScore')
    dev_datasets = [fit(dev_split, 'bertScore') for dev_split in dev_datasets]

    train_da = fit_da(train_da, 'labels')
    dev_da = fit_da(dev_da, 'labels')

    print("train size:", train.shape)
    print("dev size:", dev.shape)

    return train, dev, dev_datasets, train_da, dev_da


def get_data_loaders(trainer, model, train, train_da, seed, fold, k):
    train_chrf_df, eval_chrf_df = train_test_split(train, test_size=0.1, random_state=seed * fold + k)
    train_da_df, eval_da_df = train_test_split(train_da, test_size=0.1, random_state=seed * fold + k)
    k += 1

    # chrf + comet + bertscore
    train_chrf_examples = model.get_examples(train_chrf_df, label_type=1)

    if args.model_type == "multiple":  # add DA
        train_da_examples = model.get_examples(train_da_df, label_type=2)
        train_dataset = trainer.load_and_cache_examples(train_chrf_examples + train_da_examples, verbose=True)
    else:
        train_dataset = trainer.load_and_cache_examples(train_chrf_examples, verbose=True)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=trainer.args["dataloader_num_workers"],
    )

    # prepare eval data to evaluate results during each epoch
    chrf_eval_examples = model.get_examples(eval_chrf_df, label_type=1)
    eval_dataset = trainer.load_and_cache_examples(
        chrf_eval_examples, evaluate=True, verbose=False, silent=True
    )
    eval_chrf_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                      batch_size=VALID_BATCH_SIZE)

    da_eval_examples = model.get_examples(eval_da_df, label_type=2)
    eval_dataset = trainer.load_and_cache_examples(
        da_eval_examples, evaluate=True, verbose=False, silent=True
    )
    eval_da_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=VALID_BATCH_SIZE)

    return train_dataloader, eval_chrf_dataloader, eval_da_dataloader


def get_only_da_dataloader(train_da, seed, fold, k):
    train_da_df, eval_da_df = train_test_split(train_da, test_size=0.1, random_state=seed * fold)

    # da train data
    train_da_examples = model.get_examples(train_da_df, label_type=2)
    train_extra_finetune_dataset = trainer.load_and_cache_examples(train_da_examples, verbose=True)
    train_extra_finetune_sampler = RandomSampler(train_extra_finetune_dataset)
    train_extra_finetune_dataloader = DataLoader(
        train_extra_finetune_dataset,
        sampler=train_extra_finetune_sampler,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=trainer.args["dataloader_num_workers"],
    )

    # da eval data
    da_eval_examples = model.get_examples(eval_da_df, label_type=2)
    eval_dataset = trainer.load_and_cache_examples(
        da_eval_examples, evaluate=True, verbose=False, silent=True
    )
    eval_da_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=VALID_BATCH_SIZE)
    return train_extra_finetune_dataloader, eval_da_dataloader


if __name__ == '__main__':

    # parse arguments
    args = parse_train_args()

    # set seed
    if args.seed:
        seed = SEED_DICT[args.seed]
    else:
        seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set output dir
    if args.dir:
        if args.seed:
            TEMP_DIRECTORY = "%s/%s_%d" % (args.model_type, args.dir, args.seed)
        else:
            TEMP_DIRECTORY = "%s/%s" % (args.model_type, args.dir)
    output_dir = "/cs/labs/oabend/shachar.don/pre-translationQE/my_model/%s/outputs" % TEMP_DIRECTORY

    # print some settings
    print(sys.argv)
    print(os.environ)

    # init writer
    writer = SummaryWriter("runs/" + args.dir + "_seed_" + str(seed) + "_" + datetime.now().strftime("%m.%d.%Y, %H:%M:%S"))

    # we run the model on evaluation mode. we test it only.
    if args.test_only:
        test_only(args.test_only, args.ref, output_dir, args, epoch=args.epochs,
                  label_type=2 if (("DA" in args.test_only) or ("test20" in args.test_only)) or
                                  args.extra_finetune else 1)

    else:  # full train mode!

        # collect data
        train = get_train_data(args)
        train_da, dev_da = get_train_and_dev_da_data(args)
        dev_paths, dev_datasets, dev = get_dev_data(args)
        train, dev, dev_datasets, train_da, dev_da = fit_data(train, dev, dev_datasets, train_da, dev_da)

        # Here we will be saving the predictions
        dev_preds = np.zeros((len(dev), args.folds))
        all_devs = dev_datasets
        all_dev_preds = np.zeros((len(all_devs), len(dev), args.folds))
        comet_all_dev_preds = np.zeros((len(all_devs), len(dev), args.folds))
        bertscore_all_dev_preds = np.zeros((len(all_devs), len(dev), args.folds))
        dev_da_preds = np.zeros((1, len(dev_da), args.folds))

        # run n-folds
        for fold in range(1, args.folds + 1):
            print("fold:", fold)

            # are we on multilingual settings?
            multilingual = (args.ref if args.lang != "en" else (not args.ref)) or args.multilingual

            # init model
            # if fold > 1:
            #     model = model.detach().cpu()
            model = get_model(args.model_type, is_multilingual=multilingual, seed_to_use=seed * fold, extra_args=args)
            model, trained_epochs = load_saved_model_if_exists(model, output_dir, fold, args, extra_finetune=False)
            model.to(device)

            # init trainer object
            trainer = Trainer('roberta', "xlm-roberta-large" if multilingual else "roberta-large", num_labels=1)

            good_seed = False  # if the seed wasn't good, return!
            k = 0
            while good_seed is not True:
                if trained_epochs == args.epochs:  # we already have a good saved model
                    good_seed = True

                # init data loaders
                train_dataloader, eval_chrf_dataloader, eval_da_dataloader = get_data_loaders(trainer, model, train,
                                                                                              train_da, seed, fold, k)
                k += 1  # to manipulate the seed

                # init scheduler, optimizer
                optimizer, scheduler = init_optimizer_and_scheduler(model, total=(len(train_dataloader)
                                                                                  // GRADIENT_ACCUMULATION_STEP),
                                                                    epochs=args.epochs - trained_epochs, lr=LEARNING_RATE,
                                                                    adam_eps=ADAM_EPSILON,
                                                                    wramup_ratio=WARMUP_RATIO)

                if args.epochs <= 0:
                    print("finished initiating stuff, no epochs to run. break now")
                    break

                # train!
                last_epoch = False
                for epoch in range(1, args.epochs + 1):
                    print("epoch:", epoch, ", for fold:", fold)
                    # if loaded from existing checkpoint, jump to the current epoch
                    if trained_epochs > 0:
                        trained_epochs -= 1
                        continue

                    # run one epoch
                    if args.model_type == "multiple":
                        last_epoch = run_multitask_epoch(model, optimizer, scheduler, train_dataloader, epoch,
                                                         all_epochs=args.epochs,
                                                         eval_dataloader_chrf=eval_chrf_dataloader,
                                                         eval_dataloader_da=eval_da_dataloader)
                    else:
                        last_epoch = run_epoch(model, optimizer, scheduler, train_dataloader, epoch,
                                               all_epochs=args.epochs, eval_dataloader=eval_chrf_dataloader)

                    # save model each epoch
                    torch.save(model.state_dict(), "%s/pytorch_model_%d_%d.bin" % (output_dir, epoch, fold))

                    # save model's scores
                    report = pd.DataFrame(model.training_progress_scores)
                    report.to_csv(
                        os.path.join(output_dir, "training_progress_scores_%d.csv" % fold), index=False,
                    )

                    if last_epoch or epoch == args.epochs:
                        if epoch > 1:
                            good_seed = True
                            print("good seed! we are done. epoch:", epoch, "fold:", fold)
                        else:
                            # model = model.detach().cpu()
                            model = get_model(args.model_type, is_multilingual=multilingual,
                                              seed_to_use=seed * fold + (k + 1),
                                              extra_args=args)

                            model.to(device)
                            print("bad seed! starting the fold again. epoch:", epoch, "fold:", fold)

                        torch.save(model.state_dict(), "%s/best_model_%d.bin" % (output_dir, fold))
                        break

            # train on da only
            if args.extra_finetune > 0:

                # load saved model if exists
                model, trained_finetune_epochs = load_saved_model_if_exists(model, output_dir, fold, args,
                                                                            extra_finetune=True)
                model.to(device)

                good_seed = False  # if the seed wasn't good, return!
                if trained_finetune_epochs == args.extra_finetune:  # we already have a good saved model
                    good_seed = True
                k = 0
                while good_seed is not True:

                    # init data loaders
                    train_extra_finetune_dataloader, eval_da_dataloader = get_only_da_dataloader(train_da, seed, fold, k)

                    k += 1

                    # init scheduler, optimizer
                    optimizer_finetune, scheduler_finetune = init_optimizer_and_scheduler(model,
                                                                                          total=(len(train_extra_finetune_dataloader) // GRADIENT_ACCUMULATION_STEP),
                                                                                          epochs=args.extra_finetune - trained_finetune_epochs,
                                                                                          lr=LEARNING_RATE,
                                                                                          adam_eps=ADAM_EPSILON,
                                                                                          wramup_ratio=WARMUP_RATIO)
                    last_finetune_epoch = False

                    # clean training_progress_scores so we can save it
                    model.training_progress_scores = clean_progress_scores(model.training_progress_scores)

                    for fine_tune_epoch in range(1, args.extra_finetune + 1):
                        if trained_finetune_epochs > 0:
                            trained_finetune_epochs -= 1
                            continue

                        print("finetune epoch:", fine_tune_epoch, ", for fold:", fold)

                        if args.model_type == "multiple":
                            last_fintune_epoch = run_multitask_epoch(model, optimizer_finetune, scheduler_finetune, train_extra_finetune_dataloader,
                                      fine_tune_epoch, all_epochs=args.extra_finetune, eval_dataloader_da=eval_da_dataloader)
                        else:
                            last_fintune_epoch = run_epoch(model, optimizer_finetune, scheduler_finetune, train_extra_finetune_dataloader,
                                      fine_tune_epoch, all_epochs=args.extra_finetune, eval_dataloader=eval_da_dataloader)

                        # save model for each epoch
                        torch.save(model.state_dict(), "%s/pytorch_model_finetune_%d_%d.bin" % (output_dir, fine_tune_epoch, fold))

                        properties = ["global_step", "train_loss", "eval_loss", "pearson", "spearman"] if \
                            args.model_type != "multiple" else ["global_step", "train_loss",
                                                           "da_eval_loss", "da_pearson", "da_spearman",
                                                           "hter_eval_loss", "hter_pearson", "hter_spearman"]
                        report = {key: model.training_progress_scores[key] for key in properties}
                        report = pd.DataFrame(report)
                        report.to_csv(
                            os.path.join(output_dir, "training_progress_scores_finetune_%d.csv" % fold), index=False,
                        )

                        if last_finetune_epoch or fine_tune_epoch == args.extra_finetune:

                            if not args.search_seed:
                                print(f"we don't require good seed (fold {fold}, epoch {fine_tune_epoch}).")
                                torch.save(model.state_dict(), "%s/best_model_finetune_%d.bin" % (output_dir, fold))
                                good_seed = True
                                break

                            eval_loss, pearson, spearman, outputs, targets = evaluate(model, eval_da_dataloader,
                                                                                      fine_tune_epoch)
                            max_pearson = pearson.max()
                            print(f"pearson: {pearson}, max_pearson: {max_pearson}")

                            if max_pearson > 0.09:
                                print(f"good seed! we have done fine tuning (fold {fold}, epoch {fine_tune_epoch}).")
                                torch.save(model.state_dict(), "%s/best_model_finetune_%d.bin" % (output_dir, fold))
                                good_seed = True
                                break
                            elif fine_tune_epoch == args.extra_finetune:
                                print(f"bad seed! take another seed and start over. "
                                      f"(fold {fold}, finetune epoch {fine_tune_epoch})")
                                # model = model.detach().cpu()
                                model = get_model(args.model_type, is_multilingual=multilingual,
                                                  seed_to_use=seed * fold + k,
                                                  extra_args=args)

                                model.to(device)

            # load the last saved model for the fold

            multilingual = (args.ref if args.lang != "en" else (not args.ref)) or args.multilingual
            # model = model.detach().cpu()
            model = get_model(args.model_type, is_multilingual=multilingual,
                              seed_to_use=seed * fold + k,
                              extra_args=args)
            if args.epochs > 0:
                 model.load_state_dict(torch.load("%s/best_model_%d.bin" % (output_dir, fold)))
            model.to(device)
            model.eval()

            # use dev data to evaluate results for the i fold

            if args.model_type != "multiple":
                dev_examples = model.get_examples(dev)
                dev_dataset = trainer.load_and_cache_examples(
                    dev_examples, evaluate=True, verbose=False, silent=True
                )
                dev_sampler = SequentialSampler(dev_dataset)
                dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=VALID_BATCH_SIZE)
                dev_loss, pearson, spearman, outputs, targets_together = evaluate(model, dev_dataloader, 0)
                dev_preds[:, fold - 1] = outputs

            # split dev
            for j, dev_split in enumerate(all_devs):
                dev_examples = model.get_examples(dev_split, label_type=1)
                dev_dataset = trainer.load_and_cache_examples(
                    dev_examples, evaluate=True, verbose=False, silent=True
                )
                dev_sampler = SequentialSampler(dev_dataset)
                dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=VALID_BATCH_SIZE)

                if args.model_type == "multiple":
                    chrf_results, comet_results, bert_score_results = evaluate_multitask(model, dev_dataloader, 0, head_type=1)
                    eval_loss, pearson, spearman, outputs, targets = chrf_results
                    all_dev_preds[j, :len(dev_split), fold - 1] = outputs
                    eval_loss, pearson, spearman, outputs, targets = comet_results
                    comet_all_dev_preds[j, :len(dev_split), fold - 1] = outputs
                else:
                    dev_split_loss, pearson, spearman, outputs, targets = evaluate(model, dev_dataloader, 0)
                    all_dev_preds[j, :len(dev_split), fold - 1] = outputs

            # da dev
            dev_da_examples = model.get_examples(dev_da, label_type=2)
            dev_da_dataset = trainer.load_and_cache_examples(
                dev_da_examples, evaluate=True, verbose=False, silent=True
            )
            dev_da_sampler = SequentialSampler(dev_da_dataset)
            dev_da_dataloader = DataLoader(dev_da_dataset, sampler=dev_da_sampler, batch_size=VALID_BATCH_SIZE)

            if args.extra_finetune:
                model.load_state_dict(torch.load("%s/best_model_finetune_%d.bin" % (output_dir, fold)))
                model.to(device)
                model.eval()
            if args.model_type == "multiple":
                results_da, results_hter = evaluate_multitask(model, dev_da_dataloader, 0, head_type=2)
                eval_loss, pearson, spearman, outputs, targets = results_da
            else:
                eval_loss, pearson, spearman, outputs, targets = evaluate(model, dev_da_dataloader, 0)
            dev_da_preds[0, :len(dev_da), fold - 1] = outputs

        if args.model_type != "multiple":
            dev_preds = dev_preds.mean(axis=1)
            out_df = pd.DataFrame({'predictions': dev_preds, 'labels': targets_together})
            out_df = un_fit(out_df, 'predictions')
            out_df = un_fit(out_df, 'labels')
            pearson, spearman = print_stat_sec(out_df['predictions'].to_numpy().flatten(), out_df['labels'].to_numpy().flatten())
            out_df.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

        # split dev
        names = ["dev_%s_" % dev_name_dict[path] for path in dev_paths]
        for j, dev_split in enumerate(all_devs):
            dev_split['predictions'] = all_dev_preds[j, :len(dev_split)].mean(axis=1)
            dev_split = un_fit(dev_split, 'labels')
            dev_split = un_fit(dev_split, 'predictions')
            dev_split.to_csv(os.path.join(TEMP_DIRECTORY, names[j] + RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
            print(names[j])
            print_stat(dev_split, 'labels', 'predictions')

        # dev da
        dev_da['predictions'] = dev_da_preds[0, :len(dev_da)].mean(axis=1)
        dev_da = un_fit_da(dev_da, 'labels')
        dev_da = un_fit_da(dev_da, 'predictions')
        dev_da.to_csv(os.path.join(TEMP_DIRECTORY, "dev_da_" + RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
        print("dev_da_")
        print_stat(dev_da, 'labels', 'predictions')
