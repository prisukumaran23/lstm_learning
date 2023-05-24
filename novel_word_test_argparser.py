# LSTM novel word test argparser

import argparse

###############################################################################
# Parser for novel_word_test.py  
###############################################################################
lm_test_parser = argparse.ArgumentParser(add_help=False)
lm_test_parser.add_argument('--seed', type=int, default=1111,
                       help='random seed')
lm_test_parser.add_argument('--model_dir', type=str, default='models/',
                       help='location of the model')
lm_test_parser.add_argument('--model_name', type=str, default='m650_tied528',
                       help='name of the model')
lm_test_parser.add_argument('--mode', type=str, default='adjust_weights',
                       help='learning or adjust_weights model')
lm_test_parser.add_argument('--word1', type=str, default='table',
                       help='word 1 to combine')
lm_test_parser.add_argument('--word2', type=str, default='bureau',
                       help='word 2 to combine')
lm_test_parser.add_argument('--alpha_combine', type=float, default='0.5',
                       help='alpha for combining words with relative probabilities')
lm_test_parser.add_argument('--vocab', type=str, default='testsets/learn/',
                       help='location of the data corpus')
lm_test_parser.add_argument('--test', type=str, default='testsets/novel_nouns/',
                       help='location of the testsets') 
lm_test_parser.add_argument('--seq_len', type=float, default=20,
                       help='seq_len')
lm_test_parser.add_argument('--eval_batch_size', type=int, default=1,
                       help='eval batch size')
lm_test_parser.add_argument('--log', type=str, default='test_log.txt',
                       help='path to logging file')
lm_test_parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

###############################################################################
# Parser for few_shot_learning_maskedlm.py  
###############################################################################
lm_learning_parser = argparse.ArgumentParser(add_help=False)
lm_learning_parser.add_argument('--seed', type=int, default=1111,
                       help='random seed')
lm_learning_parser.add_argument('--model_dir', type=str, default='models/',
                       help='location of the model')
lm_learning_parser.add_argument('--init_model', type=str, default='m650_tied528_W_adjusted_',
                       help='name of the model')
lm_learning_parser.add_argument('--word1', type=str, default='table',
                       help='word 1 to combine')
lm_learning_parser.add_argument('--word2', type=str, default='bureau',
                       help='word 2 to combine')
lm_learning_parser.add_argument('--alpha_combine', type=float, default='0.5',
                       help='alpha for combining words with relative probabilities')
lm_learning_parser.add_argument('--dropout', type=float, default=0.2,
                       help='dropout applied to layers (0 = no dropout)')
lm_learning_parser.add_argument('--lr', type=float, default=20,
                       help='initial learning rate')
lm_learning_parser.add_argument('--clip', type=float, default=0.25,
                       help='gradient clipping')
lm_learning_parser.add_argument('--bptt', type=int, default=35, 
                       help='sequence length')
lm_learning_parser.add_argument('--vocab', type=str, default='testsets/learn/',
                       help='location of the data corpus')
lm_learning_parser.add_argument('--test', type=str, default='testsets/novel_nouns_rel',
                       help='location of the testsets') 
lm_learning_parser.add_argument('--test_cond', type=str, default='learn_art.adj_s_fem',
                       help='location of the data corpus')
lm_learning_parser.add_argument('--seq_len', type=float, default=20,
                       help='seq_len')
lm_learning_parser.add_argument('--eval_batch_size', type=int, default=1,
                       help='eval batch size')
lm_learning_parser.add_argument('--log', type=str, default='test_log.txt',
                       help='path to logging file')
lm_learning_parser.add_argument('--num_learning_sents', type=int, default=1,
                       help='number of learning sentences to be learnt by model')
lm_learning_parser.add_argument('--trials_learning', type=int, default=8,
                       help='number of learning & testing runs with random subset of learning sentences')
lm_learning_parser.add_argument('--no_learning', action='store_true',
                       help='learning mode off - just test adjusted weights model with no learning')
lm_learning_parser.add_argument('--mask_type', type=str, default="none",
                       help='which tokens to isolate learning for: "noun", "noun_predicate"')
lm_learning_parser.add_argument('--cuda', action='store_true',
                       help='use CUDA')
