# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dictionary_corpus
import model
import pandas as pd 
import numpy as np 
import os
import glob
import random
from utils import repackage_hidden, get_batch, batchify
from novel_word_test_argparser import lm_learning_parser

parser = argparse.ArgumentParser(parents=[lm_learning_parser],description="Few shot learning and evaluating on testsets")
args, unknown = parser.parse_known_args()

# Set the random seed manually for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("running with CUDA")
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

start = time.time()
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)
logging.info("( %.2f )" % (time.time() - start))
dictionary = dictionary_corpus.Dictionary(args.vocab)
ntokens = len(dictionary)
logging.info("Vocab size %d", ntokens)
logging.info("Torch and Cuda version:")
logging.info(torch.__version__)
logging.info(torch.version.cuda)

criterion = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()

def learn(model,learn_data):
    """Few shot learning on sentences provided in learn_data (batch of tokenized text passed through batchify function)
       Turn on training mode which enables dropout

    Args:
        model (_type_): _description_
        learn_data (_type_): batchified learning data

    Returns:
        _type_: model
    """
    model.train()
    total_loss = 0

    hidden = model.init_hidden(args.eval_batch_size)

    for i in range(0, learn_data.size(0) - 1, args.bptt):
        print("learning data size: ",learn_data.size(0))
        data, targets = get_batch(learn_data, i, args.bptt)
        #> output has size seq_length x batch_size x vocab_size (ntokens) 
        # e.g. for 5 tokens in sentence 5 X 1 X 42000
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward() # backprop the loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        print("masking LM")
        noun_idx = learn_data[-3]
        pred_idx = learn_data[-2]
        target_parameters = [p for p in model.parameters() if p.data.size(0) == ntokens]

        for p in target_parameters:
            if args.mask_type in ["noun_predicate", "noun"]:
                p.data[noun_idx].add_(p.grad.data[noun_idx], alpha=-args.lr)
                if args.mask_type == "noun_predicate":
                    p.data[pred_idx].add_(p.grad.data[pred_idx], alpha=-args.lr)
            elif args.mask_type == "none":
                p.data.add_(p.grad.data, alpha=-args.lr)
        total_loss += loss.item()
        print("total loss: ", total_loss)

    return model

def test_evaluate(model, data_source, mask, f_output):
    """_summary_

    Args:
        model (_type_): _description_
        data_source (Tensor): _description_
        mask (Tensor): _description_
        f_output (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval() # turn on evaluation mode which disables dropout
    total_loss = 0
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, args.seq_len)
            _, targets_mask = get_batch(mask, i, args.seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)
            output_candidates_probs(output_flat, targets, targets_mask,f_output)
            hidden = repackage_hidden(hidden) # detaches hidden states from their history
    return total_loss.item() / (len(data_source) - 1)

def output_candidates_probs(output_flat, targets, mask, f_output):
    """Writes output data into f_output file

    Args:
        output_flat (_type_): _description_
        targets (_type_): _description_
        mask (_type_): _description_
        f_output (_type_): _description_
    """
    log_probs = F.log_softmax(output_flat, dim=1)

    log_probs_np = log_probs.cpu().numpy()
    subset = mask.cpu().numpy().astype(bool)

    for scores, correct_label in zip(log_probs_np[subset], targets.cpu().numpy()[subset]):
        f_output.write("\t".join(str(s) for s in scores) + "\n")

def create_target_mask(test_file:str, gold_file:str, index_col):
    """_summary_

    Args:
        test_file (str): _description_
        gold_file (str): _description_
        index_col (_type_): _description_

    Returns:
        numpy.ndarray: _description_
    """
    sents = open(test_file, "r").readlines()
    golds = open(gold_file, "r").readlines()
    targets = []
    for sent, gold in zip(sents, golds):
        # gold is idx for target to be predicted as well
        # constr_id, sent_id, word_id, pos, morph
        target_idx = int(gold.split()[index_col])
        len_s = len(sent.split(" "))
        t_s = [0] * len_s   # zeroes with length of sentence, zero for each part of sentence
        t_s[target_idx] = 1 # one at target to be predicted 
        targets.extend(t_s)
    return np.array(targets)

def load_vocab(vocab_dir):
    """Loads vocabulary from file location and returns vocabulary dict

    Args:
        vocab_dir (str): location of vocabulary file

    Returns:
        dict of {str : int}: vocabulary dict with length: number of used tokens
    """
    f_vocab = open(vocab_dir, "r")
    vocab = {w: i for i, w in enumerate(f_vocab.read().split())}
    f_vocab.close()
    return vocab

def lstm_probs_out(output, gold, w2idx):
    """_summary_

    Args:
        output (list of str): list of strings of output probability for each test sentence
        gold (list of str): list of strings of index of evaluation target, correct and wrong tokens, dist between subject and target
        w2idx (dict of {str : int}): vocabulary dict (output of load_vocab)

    Returns:
        DataFrame: _description_
    """
    data = pd.DataFrame(columns=['corr_word','corr_prob','wrong_word','wrong_prob','answer']) 
    for scores, gold_info in zip(output, gold):
        scores = scores.split()
        form, form_alt = gold_info.split("\t")[1:3]

        prob_correct = float(scores[w2idx[form]])
        prob_wrong = float(scores[w2idx[form_alt]])
        if prob_correct > prob_wrong: answer_binary = 1
        else: answer_binary = 0

        data2 = pd.DataFrame([[form,prob_correct,form_alt,prob_wrong,answer_binary]],columns=['corr_word','corr_prob','wrong_word','wrong_prob','answer']) 
        data = pd.concat([data, data2],ignore_index=True)
    return data

def calc_testset(vocab_dir:str, file:str, directory_test:str, model, shuffle_idx:int, shuffle_trials:str):
    """Calculates target probabilities in test sentences for each test file and writes to output dataframe
       with model on eval mode 

    Args:
        vocab_dir (str): location of vocabulary file
        file (str): filename of test file containing test sentences
        directory_test (str): foldername of test file - same as test paradigm e.g. S_NOUN_0_ADJ 
        model (RNNModel object): model for testing either before or after learning 
        shuffle_idx (int): index of learning+testing run
        shuffle_trials (str): string of shuffled sentences learnt by model

    Returns:
        DataFrame: _description_
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    paradigm = file[len(directory_test):] # test paradigm 
    test_path = file + "/" + paradigm
    # assuming the mask file contains one number per line indicating the index of the target word
    index_col = 0

    mask = create_target_mask(test_path + ".text", test_path + ".eval", index_col)
    mask_data = batchify(torch.LongTensor(mask), args.eval_batch_size,torch.cuda.is_available())
    test_data = batchify(dictionary_corpus.tokenize(dictionary, test_path + ".text"), args.eval_batch_size,torch.cuda.is_available())

    path_output = test_path + ".output_" + "generated_" + init_model
    f_output = open(path_output, 'w')
    logging.info("Computing probabilities for model %s , %s", init_model, paradigm)
    # calculcate and write perfomance for each testset with model on evaluation mode to f_output file
    test_evaluate(model, test_data, mask_data, f_output) 
    f_output.close()

    vocab = load_vocab(vocab_dir + "/vocab.txt")
    gold = open(test_path + ".gold","r", encoding='utf8').readlines()
    info_df = pd.read_csv(test_path + '.info', sep="\t", header = None) 
    if len(info_df.columns) == 10:
        info_df.columns = ['Paradigm','Gen', 'Num', 'Sent', 'Eval', 'Corr', 'Wrong', 'Dist', 'Noun_Attr', 'Gen_Attr']
    else: info_df.columns = ['Paradigm','Gen', 'Num', 'Sent', 'Eval', 'Corr', 'Wrong', 'Dist']

    output = open(path_output).readlines()
    data_df = lstm_probs_out(output, gold, vocab)
    # insert learnt sentences into output df for information
    data_df.insert(0, "shuffle_learn_trials", shuffle_trials) 
    data_df.insert(0, "shuffle_idx", shuffle_idx)            
    # add the output probabilities from model evaluation (lstm_probs_out) to info_df which contains info on test sentences
    data_out = info_df.merge(data_df, left_index=True, right_index=True, how='inner') 
    
    percent_corr = data_df['answer'].sum()/data_df['answer'].count() # overall percent correct for all sentences
    logging.info("Percent correct for %s, %s, shuffle_idx %s : %f", paradigm, init_model, shuffle_idx, percent_corr)
    
    # delete the f_output file as its quite bulky
    if os.path.isfile(path_output):
        os.remove(path_output)
    else:
        logging.info('Error %s file not found', path_output)
    return data_out

def sampler(learning_data_dir:str):
    """Function to sample a set n (e.g. 5) sentences trial_n (.e.g 10) times from master .txt file containing possible sentences to learn 
       Why? to measure variance in performance on agreement (i.e. learning) across different sets of learning sentences

    Args:
        learning_data_dir (str): location of learn.txt file containing master list of sentences for learning

    Returns:
        list of str: list of sentences for learning 
    """
    data = open(learning_data_dir, 'r', encoding="utf8").read()
    lines = data.split("\n")
    trials = []
    while len(trials)<args.trials_learning:
        s = random.sample(lines, args.num_learning_sents)
        s_add = sorted(s)
        if sorted(s_add) not in trials:
            trials.append(s_add)
    print('Number of shuffled learning trials: ',len(trials))         
    return trials
    
def shuffle_learn():
    if args.no_learning: 
        with open(args.model_dir+init_model+".pt", 'rb') as f:
            if args.cuda:
                model = torch.load(f)
            else: # convert model trained on cuda to cpu model
                model = torch.load(f, map_location = lambda storage, loc: storage)
            logging.info("Loaded the model  %s", init_model)
            for file1 in glob.glob(os.path.join(args.test, '*')):
                for file in glob.glob(os.path.join(file1, '*')):
                    data_final = calc_testset(args.vocab, file, file1+"/", model, 0, 0)
                    paradigm = file[len(file1):]
                    test_path = file + "/" + paradigm
                    data_final.insert(0, "test_cond", file1.replace(args.test,"",1))
                    data_final.insert(0, "words_combined", args.word1+'.'+args.word2)
                    data_final.insert(0, "n_learnt", 0)
                    data_final.insert(0, "sent_learnt", 0)
                    data_final.to_csv(test_path+"."+init_model+'.csv') 
                    logging.info("Testing complete")
        del model
        logging.info("Model deleted")
    if not args.no_learning:
        trials = sampler(args.vocab+args.test_cond+".txt")
        for file1 in glob.glob(os.path.join(args.test, '*')):
            for file in glob.glob(os.path.join(file1, '*')):
                paradigm = file[len(file1):]
                test_path = file + "/" + paradigm
                # shuffle the learning sentences trials_learning times and repeat testing 
                for shuffle_idx in range(args.trials_learning):
                    random.shuffle(trials[shuffle_idx]) 
                    logging.info("shuffle id and trials: %d, %s", shuffle_idx, trials[shuffle_idx])
                    # load model and learn batchified learning sentences in learn_data 
                    with open(args.model_dir+init_model+".pt", 'rb') as f:
                        if args.cuda:
                            model = torch.load(f)
                        else: 
                            model = torch.load(f, map_location = lambda storage, loc: storage)
                        logging.info("Loaded the model  %s", init_model)
                    for trial_i in range(args.num_learning_sents):
                        print("len ",args.num_learning_sents,"i ",trial_i)
                        print(trials[shuffle_idx][trial_i])
                        global sent_learn
                        sent_learn = trials[shuffle_idx][trial_i]
                        # batchify shuffled learning sentences
                        learn_data = batchify(dictionary_corpus.tokenize_list(dictionary, [trials[shuffle_idx][trial_i]]), args.eval_batch_size, args.cuda)
                        if trial_i == 0:
                            model_learnt = learn(model,learn_data) # learning
                        else:
                            model_learnt = learn(model_learnt,learn_data) # learning
                    shuffle_trials = ','.join(trials[shuffle_idx]) # list of string: learnt sentences for logging/info
                    if shuffle_idx == 0:
                        data_final = calc_testset(args.vocab, file, file1+"/", model_learnt, shuffle_idx, shuffle_trials)
                        if shuffle_idx == args.trials_learning-1:
                            data_final.insert(0, "test_cond", file1.replace(args.test,"",1))
                            data_final.insert(0, "words_combined", args.word1+'.'+args.word2)
                            data_final.insert(0, "n_learnt", args.num_learning_sents)
                            data_final.insert(0, "sent_learnt", args.test_cond)
                            data_final.to_csv(test_path+"."+init_model+"_"+args.test_cond+"_"+str(args.num_learning_sents)+'.csv') 
                            logging.info("Testing complete")
                    elif shuffle_idx < args.trials_learning-1:
                        df = calc_testset(args.vocab, file, file1+"/", model_learnt, shuffle_idx, shuffle_trials)
                        data_final = pd.concat([data_final, df])
                        logging.info("Adding outputs of shuffle run id:  %d", shuffle_idx)
                    else:
                        df = calc_testset(args.vocab, file, file1+"/", model_learnt, shuffle_idx, shuffle_trials)
                        data_final = pd.concat([data_final, df])
                        logging.info("Adding outputs of shuffle run id:  %d", shuffle_idx)
                        data_final.insert(0, "test_cond", file1.replace(args.test,"",1))
                        data_final.insert(0, "words_combined", args.word1+'.'+args.word2)
                        data_final.insert(0, "n_learnt", args.num_learning_sents)
                        data_final.insert(0, "sent_learnt", args.test_cond)
                        data_final.to_csv(test_path+"."+init_model+"_"+args.test_cond+"_"+str(args.num_learning_sents)+'.csv') 
                        logging.info("Testing complete")
                    del model
                    del model_learnt

sent_learn = ""
init_model = args.init_model+args.word1+"."+args.word2

shuffle_learn()
