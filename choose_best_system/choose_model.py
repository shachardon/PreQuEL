import os
os.environ['TRANSFORMERS_CACHE'] = '/cs/labs/oabend/shachar.don/pre-translationQE/hg_cache'
import sys
sys.path.append("/cs/labs/oabend/shachar.don/pre-translationQE/my_model/")

import csv
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import argparse
import gc
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from shared_utils import parse_train_args, SEED_DICT, load_saved_model_if_exists, init_optimizer_and_scheduler
from systemsDataset import SystemsDataset, read_annotated_file
from choose_model_classification import ChooseModel
from choose_model_regression import ChooseModelReg
import math
from transformers import get_linear_schedule_with_warmup


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Set pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-5
ADAM_EPSILON = 1e-8
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEP = 1
SEED = 777
TEMP_DIRECTORY = "try"

SYSTEMS_DICT = {"m2m_100_418m": "/cs/snapless/oabend/shachar.don/data/data_for_marian/m2m_100_418m/",
                "mbart50_m2m": "/cs/snapless/oabend/shachar.don/data/data_for_marian/mbart50_m2m/",
                "m2m_100_12b": "/cs/snapless/oabend/shachar.don/data/data_for_marian/m2m_100_1.2B/",
                "marian": "/cs/snapless/oabend/shachar.don/data/data_for_marian/"}

DATA_DICT_MARIAN = {"Tatoeba": "Tatoeba.de-en.chrf.ngram.lan.comet.bertScore",
             "wmt": "WMT-News.de-en.chrf.ngram.lan.comet.bertScore",
             "news20": "newstest2020.ende.chrf.ngram.lan.comet.bertScore",
             "bible": "bible-uedin.de-en.chrf.ngram.lan.comet.bertScore",
             "global": "GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore"}

DATA_DICT = {"Tatoeba": "Tatoeba.de-en.chrf.comet.bertScore",
             "wmt": "WMT-News.de-en.chrf.comet.bertScore",
             "news20": "newstest2020.ende.chrf.comet.bertScore",
             "bible": "bible-uedin.de-en.chrf.comet.bertScore",
             "global": "GlobalVoices.de-en.chrf.comet.bertScore"}


def get_train_data_for_data_and_system(data, system, args, seed):
    system_name = SYSTEMS_DICT[system]
    if system == "marian":
        path = system_name + DATA_DICT_MARIAN[data] + ".train"
    else:
        path = system_name + DATA_DICT[data] + ".train"
    df_train = read_annotated_file(path, chrf=True if "chrf" in args.score_types else False,
                                   comet=True if "comet" in args.score_types else False,
                                   bertScore=True if "bertScore" in args.score_types else False)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=seed)

    return df_train, df_val


def get_dev_data_for_data_and_system(data, system, args):
    system_name = SYSTEMS_DICT[system]
    if system == "marian":
        path = system_name + DATA_DICT_MARIAN[data] + ".dev"
    else:
        path = system_name + DATA_DICT[data] + ".dev"
    df_dev = read_annotated_file(path, chrf=True if "chrf" in args.score_types else False,
                                 comet=True if "comet" in args.score_types else False,
                                 bertScore=True if "bertScore" in args.score_types else False)

    return df_dev


def get_train_dataloders(args, seed):
    train = []
    val = []

    for i, system in enumerate(args.systems):
        system_train = []
        system_val = []
        for j, data in enumerate(args.data):
            df_train, df_val = get_train_data_for_data_and_system(data, system, args, seed)
            system_train.append(df_train)
            system_val.append(df_val)
        train.append(pd.concat(system_train).reset_index(drop=True))
        val.append(pd.concat(system_val).reset_index(drop=True))

    train_dataset = SystemsDataset(train, args.score_types)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    val_dataset = SystemsDataset(val, args.score_types)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_dev_dataloders(args):
    split_devs = [[] for i in range(len(args.data))]

    for i, data in enumerate(args.data):
        for j, system in enumerate(args.systems):
            df_dev = get_dev_data_for_data_and_system(data, system, args)
            split_devs[i].append(df_dev.reset_index(drop=True))

    dev_datasets = [SystemsDataset(split_dev, args.score_types) for split_dev in split_devs]
    dev_dataloaders = [DataLoader(dev_dataset, batch_size=4, shuffle=False, num_workers=0)
                       for dev_dataset in dev_datasets]

    return dev_dataloaders


def calcuate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct


def run_epoch(model, epoch, train_dataloader, val_dataloader, optimizer, scheduler, args):
    if args.model_type == "choose_reg":
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    steps_in_epoch = len(train_dataloader)
    if steps_in_epoch > 10000:
        run_val = 3000
    else:
        run_val = 300

    best_loss = -1
    counter = 0
    # last_epoch = False

    model.train()
    for _, data in tqdm(enumerate(train_dataloader, 0), desc=f"Running Epoch {epoch} of {args.epochs}"):

        optimizer.zero_grad()

        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['input_mask'].to(device, dtype=torch.long)
        token_type_ids = data['segment_ids'].to(device, dtype=torch.long)
        targets = data['best_%s' % args.score_types[0]].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print("train - targets:", targets, "outputs:", outputs)
        # print("train - targets:", targets, "outputs.logits:", outputs.logits)

        if args.model_type == "choose_reg":
            loss = 0
            for i in range(len(args.systems)):
                system_target = data['%s_%d' % (args.score_types[0], i)]
                system_target = system_target.to(device, dtype=torch.float)
                # print("train - targets:", system_target, "outputs:", outputs[i])
                loss += loss_func(outputs[i].view(-1), system_target.view(-1))

            tr_loss += loss.item()
            big_val, big_idx = torch.max(torch.cat(outputs, dim=1), dim=1)
        else:
            loss = loss_func(outputs.logits, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.logits, dim=1)

        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % run_val == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per {run_val} steps: {loss_step}")
            print(f"Training Accuracy per {run_val} steps: {accu_step}")

            model.training_progress_scores["global_step"].append((steps_in_epoch * (epoch - 1)) + nb_tr_steps)
            model.training_progress_scores["train_loss"].append(loss_step)
            model.training_progress_scores["train_accuracy"].append(accu_step)

            if val_dataloader:
                print("\neval")
                val_loss, val_acc = evaluate(model, epoch, val_dataloader, args)
                model.training_progress_scores["eval_loss"].append(val_loss)
                model.training_progress_scores["eval_accuracy"].append(val_acc)

                if best_loss < 0 or best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), "%s/best_model.bin" % output_dir)
                    counter = 0
                else:
                    counter += 1
                    print("model didn't improve for", counter, "eval steps")
                    if counter > 9:
                        # last_epocch = True
                        print("exiting training")
                        break
            print(model.training_progress_scores)


        loss.backward()
        # # When using GPU
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()

    scheduler.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    model.training_progress_scores["global_step"].append((steps_in_epoch * (epoch - 1)) + nb_tr_steps)
    model.training_progress_scores["train_loss"].append(epoch_loss)
    model.training_progress_scores["train_accuracy"].append(epoch_accu)
    if val_dataloader:
        val_loss, val_acc = evaluate(model, epoch, val_dataloader, args)
        model.training_progress_scores["eval_loss"].append(val_loss)
        model.training_progress_scores["eval_accuracy"].append(val_acc)

    model.load_state_dict(torch.load("%s/best_model.bin" % output_dir))  # we want to finish the epoch with the best model loaded.


def run_epoch_reg(model, epoch, train_dataloader, val_dataloader, optimizer, scheduler, args):
    loss_func = torch.nn.MSELoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    steps_in_epoch = len(train_dataloader)
    if steps_in_epoch > 10000:
        run_val = 3000
    else:
        run_val = 300

    best_loss = -1
    counter = 0
    # last_epoch = False

    model.train()
    for _, data in tqdm(enumerate(train_dataloader, 0), desc=f"Running Epoch {epoch} of {args.epochs}"):

        optimizer.zero_grad()

        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['input_mask'].to(device, dtype=torch.long)
        token_type_ids = data['segment_ids'].to(device, dtype=torch.long)
        # targets = data['all_comet'].to(device, dtype=torch.float)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print("train - targets:", targets, "outputs:", outputs)

        loss = 0
        for i in range(len(args.systems)):
            system_target = data['%s_%d' % (args.score_types[0], i)]
            system_target = system_target.to(device, dtype=torch.float)
            # print("train - targets:", system_target, "outputs:", outputs[i])
            loss += loss_func(outputs[i].view(-1), system_target.view(-1))

        tr_loss += loss.item()
        big_val, big_idx = torch.max(torch.cat(outputs, dim=1), dim=1)
        n_correct += calcuate_accuracy(big_idx, data["best_%s" % args.score_types[0]].to(device, dtype=torch.long))
        nb_tr_steps += 1
        nb_tr_examples += system_target.size(0)

        if _ % run_val == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per {run_val} steps: {loss_step}")
            print(f"Training Accuracy per {run_val} steps: {accu_step}")

            model.training_progress_scores["global_step"].append((steps_in_epoch * (epoch - 1)) + nb_tr_steps)
            model.training_progress_scores["train_loss"].append(loss_step)
            model.training_progress_scores["train_accuracy"].append(accu_step)

            if val_dataloader:
                print("\neval")
                val_loss, val_acc = evaluate_reg(model, epoch, val_dataloader, args)
                model.training_progress_scores["eval_loss"].append(val_loss)
                model.training_progress_scores["eval_accuracy"].append(val_acc)

                if best_loss < 0 or best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), "%s/best_model.bin" % output_dir)
                    counter = 0
                else:
                    counter += 1
                    print("model didn't improve for", counter, "eval steps")
                    if counter > 9:
                        # last_epocch = True
                        print("exiting training")
                        break
            print(model.training_progress_scores)


        loss.backward()
        # # When using GPU
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()

    scheduler.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    model.training_progress_scores["global_step"].append((steps_in_epoch * (epoch - 1)) + nb_tr_steps)
    model.training_progress_scores["train_loss"].append(epoch_loss)
    model.training_progress_scores["train_accuracy"].append(epoch_accu)
    if val_dataloader:
        val_loss, val_acc = evaluate_reg(model, epoch, val_dataloader, args)
        model.training_progress_scores["eval_loss"].append(val_loss)
        model.training_progress_scores["eval_accuracy"].append(val_acc)

    model.load_state_dict(torch.load("%s/best_model.bin" % output_dir))  # we want to finish the epoch with the best model loaded.


def evaluate(model, epoch, eval_dataloader, args):
    if args.model_type == "choose_reg":
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    run_val = 200

    model.eval()

    best_system_grade = 0
    oracle_grade = 0
    pred_grade = 0

    for _, data in tqdm(enumerate(eval_dataloader, 0), desc=f"Running Epoch {epoch} of {args.epochs}"):
        with torch.no_grad():

            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)
            token_type_ids = data['segment_ids'].to(device, dtype=torch.long)
            targets = data['best_%s' % args.score_types[0]].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            # print("train - targets:", targets, "outputs:", outputs)

            if args.model_type == "choose_reg":
                loss = 0
                for i in range(len(args.systems)):
                    system_target = data['%s_%d' % (args.score_types[0], i)]
                    system_target = system_target.to(device, dtype=torch.float)
                    # print("train - targets:", system_target, "outputs:", outputs[i])
                    loss += loss_func(outputs[i].view(-1), system_target.view(-1))

                tr_loss += loss.item()
                big_val, big_idx = torch.max(torch.cat(outputs, dim=1), dim=1)

                for i in range(len(targets)):
                    oracle_grade += data['%s_%d' % (args.score_types[0], targets[i])][i]
                    pred_grade += data['%s_%d' % (args.score_types[0], big_idx[i])][i]
                best_system_grade += data['%s_%d' % (args.score_types[0], 3)].sum()  # marian

            else:
                loss = loss_func(outputs.logits, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.logits, dim=1)

            n_correct += calcuate_accuracy(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % run_val == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per {run_val} steps: {loss_step}")
                print(f"Validation Accuracy per {run_val} steps: {accu_step}")
                print(f"Validation Oracle Grade Epoch: {oracle_grade / nb_tr_examples}")
                print(f"Validation Pred Grade Epoch: {pred_grade / nb_tr_examples}")
                print(f"Validation Best-System Grade Epoch: {best_system_grade / nb_tr_examples}")

            gc.collect()
            torch.cuda.empty_cache()

    # scheduler.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    print(f"Validation Oracle Grade Epoch: {oracle_grade / nb_tr_examples}")
    print(f"Validation Pred Grade Epoch: {pred_grade / nb_tr_examples}")
    print(f"Validation Best-System Grade Epoch: {best_system_grade / nb_tr_examples}")

    return epoch_loss, epoch_accu


def evaluate_reg(model, epoch, eval_dataloader, args):
    loss_func = torch.nn.MSELoss()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    run_val = 200

    model.eval()
    for _, data in tqdm(enumerate(eval_dataloader, 0), desc=f"Running Epoch {epoch} of {args.epochs}"):
        with torch.no_grad():

            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)
            token_type_ids = data['segment_ids'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            # print("train - targets:", targets, "outputs:", outputs)

            loss = 0
            for i in range(len(args.systems)):
                system_target = data['%s_%d' % (args.score_types[0], i)]
                system_target = system_target.to(device, dtype=torch.float)
                # print("train - targets:", system_target, "outputs:", outputs[i])
                loss += loss_func(outputs[i].view(-1), system_target.view(-1))

            tr_loss += loss.item()
            big_val, big_idx = torch.max(torch.cat(outputs, dim=1), dim=1)
            n_correct += calcuate_accuracy(big_idx, data["best_%s" % args.score_types[0]].to(device, dtype=torch.long))

            nb_tr_steps += 1
            nb_tr_examples += system_target.size(0)

            if _ % run_val == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per {run_val} steps: {loss_step}")
                print(f"Validation Accuracy per {run_val} steps: {accu_step}")

            gc.collect()
            torch.cuda.empty_cache()

    # scheduler.step()  # ?

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    return epoch_loss, epoch_accu


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
        pass

    else:  # full train mode!
        dev_dataloaders = get_dev_dataloders(args)  # the same one for all folds.

        for fold in range(1, args.folds + 1):
            k = 0
            good_fold = False

            if args.model_type == "choose_reg":
                model = ChooseModelReg(seed, len(args.systems))
            else:
                model = ChooseModel(seed, len(args.systems))

            model, trained_epochs = load_saved_model_if_exists(model, output_dir, fold, args, extra_finetune=False)
            model.to(device)

            while not good_fold:
                good_fold = True  # for now. maybe change later.
                train_dataloader, val_dataloader = get_train_dataloders(args, seed * fold + k)

                # init scheduler, optimizer
                optimizer, scheduler = init_optimizer_and_scheduler(model, total=(len(train_dataloader)
                                                                                  // GRADIENT_ACCUMULATION_STEP),
                                                                    epochs=args.epochs - trained_epochs,
                                                                    lr=LEARNING_RATE,
                                                                    adam_eps=ADAM_EPSILON,
                                                                    wramup_ratio=WARMUP_RATIO)

                # for batch in train_dataloader:
                #     print(batch)
                for epoch in range(1, args.epochs + 1):

                    if trained_epochs > 0:
                        trained_epochs -= 1
                        continue

                    run_epoch(model, epoch, train_dataloader, val_dataloader, optimizer, scheduler, args)

                    # save model's scores
                    report = pd.DataFrame(model.training_progress_scores)
                    report.to_csv(
                        os.path.join(output_dir, "training_progress_scores_%d.csv" % fold), index=False,
                    )
                    # save the model each epoch
                    torch.save(model.state_dict(), "%s/pytorch_model_%d_%d.bin" % (output_dir, epoch, fold))
                k += 1
