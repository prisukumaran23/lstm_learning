# LSTM novel word test main

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import pandas as pd 
import numpy as np 
import dictionary_corpus
import data
from utils import repackage_hidden, batchify, get_batch
import os
import glob
import logging
import argparse
from novel_word_test_argparser import lm_test_parser

###############################################################################
# Load args and start logging 
###############################################################################

parser = argparse.ArgumentParser(parents=[lm_test_parser],description="Evaluating RNN LM on testsets")

args, unknown = parser.parse_known_args()
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

###############################################################################
# Testing code
###############################################################################

def find_key(input_dict, value):
    result = "None"
    for key,val in input_dict.items():
        if val == value:
            result = key
    return result

def get_embedding(word_list):
    weights = model.encoder.weight
    emb_list = []
    for w in word_list['word']:
        idx = word_list['idx'].loc[word_list['word'] == w].values[0]
        word_embedding = weights[int(idx)].detach().numpy().tolist()
        emb_list.append(word_embedding)
        dict_noun = word_list.to_dict('list')
    dict_noun['embedding'] = emb_list
    return dict_noun

def new_weights():
    # combine weights for two semantically similar words of opposite target feature (e.g. gender/number)
    # replace weights for last token in vocab with novel word 
    # use new set of weights in an adjusted model for few shot learning tests
    weights = model.encoder.weight
    word2idx = data.Dictionary(args.vocab+"vocab.txt").word2idx
    print(args.word1, args.word2)
    combine = {'word': [args.word1, args.word2],
    'Gender': ['F', 'M'],
    'Number': ['S', 'S'], 
    'Article': ['la', 'le']}
    df_combine = pd.DataFrame(combine)
    for w in combine['word']:
        df_combine.loc[df_combine['word'] == w, 'idx'] = int(word2idx[w])
    l = args.alpha_combine
    print(args.alpha_combine)
    df_combine = pd.DataFrame(get_embedding(df_combine))
    emb1 = [i * l for i in df_combine['embedding'][0]]
    emb2 = [i * (1-l) for i in df_combine['embedding'][1]]
    new_emb = [sum(a) for a in zip(emb1, emb2)]
    new_emb_tensor = torch.FloatTensor(new_emb)
    new_weights = weights
    with torch.no_grad():
        new_weights[42907] = new_emb_tensor
    return new_weights

def evaluate(data_source, mask, f_output):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            _, targets_mask = get_batch(mask, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)

            output_candidates_probs(output_flat, targets, targets_mask,f_output)

            hidden = repackage_hidden(hidden) # detaches hidden states from their history

    return total_loss.item() / (len(data_source) - 1)

def adjust_weights_and_evaluate(data_source, mask, f_output):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.encoder.weight = new_weights()
    model.decoder.weight = new_weights()
    total_loss = 0

    hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            _, targets_mask = get_batch(mask, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)

            output_candidates_probs(output_flat, targets, targets_mask,f_output)

            hidden = repackage_hidden(hidden) # detaches hidden states from their history

    return total_loss.item() / (len(data_source) - 1)

def output_candidates_probs(output_flat, targets, mask, f_output):
    log_probs = F.log_softmax(output_flat, dim=1)

    log_probs_np = log_probs.cpu().numpy()
    subset = mask.cpu().numpy().astype(bool)

    for scores, correct_label in zip(log_probs_np[subset], targets.cpu().numpy()[subset]):
        f_output.write("\t".join(str(s) for s in scores) + "\n")

def create_target_mask(test_file, gold_file, index_col):
    sents = open(test_file, "r").readlines()
    golds = open(gold_file, "r").readlines()
    #TODO optimize by initializaing np.array of needed size and doing indexing
    targets = []
    for sent, gold in zip(sents, golds):
        # gold is idx for target to be predicted as well
        target_idx = int(gold.split()[index_col])
        len_s = len(sent.split(" "))
        t_s = [0] * len_s # zeroes with length of sentence, zero for each part of sentence
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
    data = pd.DataFrame(columns=['corr_word', 'corr_prob', 'wrong_word','wrong_prob','answer']) 
    for scores, gold_info in zip(output, gold):
        scores = scores.split()
        form, form_alt = gold_info.split("\t")[1:3]

        prob_correct = float(scores[w2idx[form]])
        prob_wrong = float(scores[w2idx[form_alt]])
        if prob_correct > prob_wrong: answer_binary = 1
        else: answer_binary = 0

        data2 = pd.DataFrame([[form,prob_correct,form_alt,prob_wrong,answer_binary]],columns=['corr_word', 'corr_prob', 'wrong_word','wrong_prob','answer']) 
        data = pd.concat([data, data2],ignore_index=True)
    return data

def calc_testset(data_dir, directory_test, model_name):
    """Calculates target probabilities in test sentences for each test file and writes to output dataframe
       with model on eval mode 

    Args:
        data_dir (str): location of vocabulary file
        directory_test (str): folder containing test files  
        model_name (str): model for testing either before or after learning 

    Returns:
        DataFrame
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    data_summary = pd.DataFrame(columns=('Paradigm', 'Model', 'Percent_Corr'))
    for file in glob.glob(os.path.join(directory_test, '*')):
        paradigm = file[len(directory_test):]
        test_path = file + "/" + paradigm
        
        # assuming the mask file contains one number per line indicating the index of the target word
        index_col = 0

        mask = create_target_mask(test_path + ".text", test_path + ".eval", index_col)
        mask_data = batchify(torch.LongTensor(mask), args.eval_batch_size,torch.cuda.is_available())
        test_data = batchify(dictionary_corpus.tokenize(dictionary, test_path + ".text"), args.eval_batch_size,torch.cuda.is_available())

        path_output = test_path + ".output_" + "generated_" + model_name
        f_output = open(path_output, 'w')
        logging.info("Computing probabilities for model %s , %s", model_name, paradigm)
        if learning_mode == 'adjust_weights':
            adjust_weights_and_evaluate(test_data, mask_data, f_output)
        else: evaluate(test_data, mask_data, f_output)
        f_output.close()

        vocab = load_vocab(data_dir + "/vocab.txt")
        gold = open(test_path + ".gold","r", encoding='utf8').readlines()
        info_df = pd.read_csv(test_path + '.info', sep="\t", header = None)
        if len(info_df.columns) == 10:
            info_df.columns = ['Paradigm','Gen', 'Num', 'Sent', 'Eval', 'Corr', 'Wrong', 'Dist', 'Noun_Attr', 'Gen_Attr']
        else: info_df.columns = ['Paradigm','Gen', 'Num', 'Sent', 'Eval', 'Corr', 'Wrong', 'Dist']

        output = open(path_output).readlines()
        data_df = lstm_probs_out(output, gold, vocab)
        
        data_out = info_df.merge(data_df, left_index=True, right_index=True, how='inner')
        if args.alpha_combine != 0.5:
            data_out.to_csv(test_path+"."+model_name+"_W_adjusted_"+args.word1+'.'+args.word2+"_alpha"+str(args.alpha_combine)+'.csv') 
        else:
            data_out.to_csv(test_path+"."+model_name+"_W_adjusted_"+args.word1+'.'+args.word2+'.csv') 

        percent_corr = data_df['answer'].sum()/data_df['answer'].count()
        logging.info("Percent correct for %s, %s : %f", paradigm, model_name, percent_corr)
        df = pd.DataFrame({'Paradigm':paradigm,'Model':model_name, 'Percent_Corr':percent_corr}, index=[0])
        data_summary = data_summary.append(df)
        
        # Delete the output gen file as its quite bulky
        if os.path.isfile(path_output):
            os.remove(path_output)
        else:   
            logging.info('Error %s file not found', path_output)

    if not os.path.exists(directory_test + '.csv'):
        data_summary.to_csv(directory_test + '.csv')
    data_summary.to_csv(directory_test + '.csv', mode='a', index=False, header=False)
    logging.info('Test complete')

###############################################################################
# Load the model
###############################################################################
learning_mode = args.mode

model_file = args.model_dir + args.model_name + ".pt"
print('loaded model ', args.model_name)
with open(model_file, 'rb') as f:
    logging.info('Loading the model ' + model_file)
    if args.cuda:
            model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = args.eval_batch_size
seq_len = args.seq_len

dictionary = dictionary_corpus.Dictionary(args.vocab)
vocab_size = len(dictionary)

###############################################################################
# Test model in eval mode
###############################################################################
# Calc perfomance for each testset
for file in glob.glob(os.path.join(args.test, '*')):
    calc_testset(args.vocab, file+"/", args.model_name)

###############################################################################
# Save model with adjusted weight matrix replacing last token in vocab
###############################################################################
if args.mode == 'adjust_weights':
    print('saving model ', args.model_name)
    torch.save(model, args.model_dir+args.model_name+'_W_adjusted_'+args.word1+'.'+args.word2+'.pt')