#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:12:14 2019

@author: JenniferLiu
"""

import argparse
import pprint
import string
import re
import math

import nltk
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LaplaceProbDist
from nltk.probability import WittenBellProbDist
from nltk.probability import LidstoneProbDist
from nltk.probability import UniformProbDist
from nltk import tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
pp = pprint.PrettyPrinter(indent=4)
nltk.download('stopwords')

class DecisionList(object):

    def __init__(self, train_file, test_file=None):
        """
        Implementation of Supervised Yarowsky Decision list
        for the homograph disambiguation problem. 
        Constructor takes as input a string of test data of the form:
            *word context of word to train the classifier on
            word  context of the word with a different sense 
        """
        
        
        # Word with or without *
        self.word = train_file.readline().split(":")[0]  # + log-likelihood == this sense
        self.word_star = "*" + self.word   # - log-likelihood == this sense 
    
        # train data
        self.train = self.clean_corpus(train_file)
        
        # Testing data
        if test_file:
            self.test = self.clean_corpus(test_file)
    
        # ConditionalFreqDist - total frequencies 
        self.cfd = None
        
        # ConditionalProbDist 
        self.cpd = None
        
        # Store the decision list
        self.decisionList = [] 
        
        # Store all results in this dictionary
        # Accuracy, Error Rate etc... 
        self.res = dict()
    
    def clean_corpus(self, file):
        """
        Clean out the corpus. 
        """
        format_corpus = []
        
        for text in file:
            # split the text into its individual senses and contexts
            corpus = text.split("\n")
            # split the sense from the context
            corpus = [l.split("\t") for l in corpus if l != '']
            # strip the colon from the sense
            corpus = [[l[0][:-1], l[1]] for l in corpus]
            # remove XML tags from corpus
            corpus = [[l[0], re.sub(r'\<.*?(\>|$)', '', l[1])] for l in corpus]
            # Punkt tokenize the context
            corpus = [[l[0], tokenizer.tokenize(l[1].lower())] for l in corpus]
            # Get rid of stop words and punctuation from the context
            stop_words = stopwords.words("english")
            stop_words.extend(string.punctuation)
            # Get pos tags, but store them in adjacent array because it makes
            # root sense lookup easier
            corpus = [[l[0], [w for w in tag.pos_tag(l[1])]] for l in corpus]
            # Remove punctuation from words
            corpus = [[l[0], [(re.sub(r'[\.\,\?\!\'\"\-\_]','', w[0]), w[1]) for w in l[1]]] for l in corpus]
            # only keep context words that aren't in our stop words list and that
            # are shorter than two characters long
            corpus = [[l[0], [w for w in l[1] if w[0] not in stop_words and len(w[0]) > 1]] for l in corpus]
    
            # Change the structure of the corpus        
            new_format = [corpus[0][0], [], []]
            
            for i in range(len(corpus[0][1])):
                #print(corpus[0][1][i], end="")
                w_tmp, p_tmp = corpus[0][1][i][0], corpus[0][1][i][1]
                new_format[1].append(w_tmp)
                new_format[2].append(p_tmp)
            
            format_corpus.append(new_format)
        
        return format_corpus
    
    def fit(self):
        """
        Generate the decision list for the training set
        """
        self.get_dist()
        self.gen_decisionList()
    
    def get_dist(self, smooth=None):
        """
        Creates the conditional freq and prob dist to
        generate decision list rules
        """
        self.cfd = ConditionalFreqDist()
        
        self.k_window_dist(self.train, 5)
        self.k_word_dist(self.train, 1)
        self.k_word_dist(self.train, -1)
        self.k_tag_dist(self.train, 1)
        self.k_tag_dist(self.train, -1)
        
        if smooth:
            self.cpd = ConditionalProbDist(self.cfd, smooth)
        else:
            self.cpd = ConditionalProbDist(self.cfd, LidstoneProbDist, 0.1)
    
    def gen_decisionList(self):
        """
        Generate decision list 
        """
    
        for rule in self.cpd.conditions():
            log_l = self.compute_log_likelihood(rule)
            self.decisionList.append([rule, log_l])
        
        self.decisionList.sort(key=lambda key: math.fabs(key[1]), reverse=True)
        #pp.pprint(self.decisionList)
        
    def compute_log_likelihood(self, rule):
        """
        Compute the log likelihood
        """
        word_p = self.cpd[rule].prob(self.word)
        word_star_p = self.cpd[rule].prob(self.word_star)
        div = word_p / word_star_p
        
        # neg = word_star_p , pos == word_p
        if div == 0:
            return 0
        else:
            return math.log2(div)
    
    def get_k_word(self, k, context):
        word = context.index(self.word)
        k_word_i = word + k 
        if len(context) > k_word_i and k_word_i >= 0:
            return context[k_word_i]
        else:
            return False 
    
    def get_k_tag(self, k, context, tags):
        word = context.index(self.word)
        k_tag_i = word + k
        if len(context) > k_tag_i and k_tag_i >= 0:
            return tags[k_tag_i]
        else:
            return False 
     
    def k_window_dist(self, corpus, k):
        """
        Generate a rule for every word seen in the window 
        Generate it up to the kth word (left and right) in the corpus 
        """
        for line in corpus:
            k_ = k
            sense, context = line[0], line[1]
            while k_ > 0:
                pos_k_word = self.get_k_word(k_, context)
                neg_k_word = self.get_k_word(-1 * k_, context)
                if pos_k_word:
                    condition = str(k)+"_window_"+re.sub(r'\_', '', pos_k_word)
                    self.cfd[condition][sense] += 1
                if neg_k_word:
                    condition = str(k)+"_window_"+re.sub(r'\_', '', neg_k_word)
                    self.cfd[condition][sense] += 1
                k_ -= 1
    
    def k_word_dist(self, corpus, k):
        for line in corpus:
            sense, context = line[0], line[1]
            k_word = self.get_k_word(k, context)
            if k_word:
                # create freqdist for each sense per word
                condition = str(k) + "_word_" + re.sub(r'\_', '', k_word)
                self.cfd[condition][sense] += 1
    
    def k_tag_dist(self, corpus, k):
        for line in corpus:
            sense, context, tag = line[0], line[1], line[2]
            k_tag = self.get_k_tag(k, context, tag)
            if k_tag:
                # create freqdist for each sense per word
                condition = str(k) + "_tag_" + re.sub(r'\_', '', k_tag)
                self.cfd[condition][sense] += 1
    
    
    # Testing the decision list 
    def evaluate(self, test_file=None):
        if test_file:
            self.test = self.clean_corpus(test_file)
        
        word_prior, word_star_prior = 0.0, 0.0
        
        # discover how many times word occured
        # and # times *word occured
        for data in self.train:
            if data[0] == self.word:
                word_prior += 1
            elif data[0] == self.word_star:
                word_star_prior += 1
            else:
                print("warning no match")

        # total data points 
        self.res["total"] = word_prior + word_star_prior
        
        
        # word prior
        self.res["word_prior"] = word_prior/self.res["total"]
        
        # *word prior
        self.res["word_star_prior"] = word_star_prior/self.res["total"]
        
        # Determine which has the higher word prior 
        if self.res["word_star_prior"] > self.res["word_prior"]:
            self.majority_label = self.word_star
            self.res["prior_prob"] = self.res["word_star_prior"]
        else:
            self.majority_label = self.word
            self.res["prior_prob"] = self.res["word_prior"]
        
        # Process test_data
        if self.test:
            self.res["TP"] = 0 # num of Predicted = Word, True sense = Word
            self.res["TN"] = 0 # num of Predicted = *word, True sense = *word
            self.res["FP"] = 0 # num of Predicted = word, True sense = *word
            self.res["FN"] = 0 # num of Predicted = *word, True sense = word 
            
            self.res["correct"] = []
            self.res["incorrect"] = [] 
            
            # Total # of test data points 
            total_test = 0
            
            
            # Start predicting 
            for data in self.test:
                predicted, true_sense, rule, context = self.predict(data)
                
                total_test += 1
                
                if predicted == true_sense:
                    self.res["correct"].append([predicted, true_sense, rule, context])                        
                    
                    if predicted == self.word and true_sense == self.word:
                        self.res["TP"] += 1
                    elif predicted == self.word_star and true_sense == self.word_star:
                        self.res["TN"] += 1
                    
                elif predicted != true_sense:
                    self.res["incorrect"].append([predicted, true_sense, rule, context])
                    
                    if predicted == self.word and true_sense == self.word_star:
                        self.res["FP"] += 1
                    elif predicted == self.word_star and true_sense == self.word:
                        self.res["FN"] += 1
            
            
            self.res["accuracy"] = len(self.res["correct"])/total_test
            
            # Compute % error reduction over baseline
            self.res["error"] = 1 - self.res["accuracy"]
            self.res["baseline_error"] = 1 - self.res["prior_prob"]
            
            # 1 - (error from test data/baseline error)
            self.res["error_reduction"] = 1 - self.res["error"]/self.res["baseline_error"]
    
    def print_eval(self):
        print("Total {}".format(self.res["total"]))
        print("Word Prior: {}".format(self.res["word_prior"]))
        print("*Word Prior: {}".format(self.res["word_star_prior"]))
        
        print("{:<30}{:>.3%}"
                .format("Majority Class Prior Prob: ",
                   self.res["prior_prob"]))
        print("{:<30}{:>}"
                .format("Majority Class Label: ", self.majority_label))
        
        print()
        print("Top 10 Rules:")
        for d in self.decisionList[:10]:
            print("{:<30}{:>.4}".format(d[0], d[1]))
        
        print()
        print("{:<30}{:>.3%}"
                .format("Accuracy: ", self.res["accuracy"]))
        
        print()
        print("Confusion Matrix Stats")
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(self.res["TP"], self.res["TN"], self.res["FP"], self.res["FN"]))
        
        print()
        print("3 Correct Predictions:")
        for correct in self.res["correct"][:3]:
            predicted, true_sense, rule, context = correct[0], correct[1], correct[2], correct[3]
            
            if rule == "default":
                print("Correctly Predicted: {} \n Rule: {}, \n Sentence: {}"
                        .format(predicted, "Used majority baseline", " ".join(context)))
            else:
                print("Correctly Predicted: {} \n Rule: {}, log-likelihood: {:.3f} \n Sentence: {}"
                        .format(predicted, rule[0], rule[1], " ".join(context)))
    
        print()
        print("3 Incorrect Predictions:")
        for correct in self.res["incorrect"][:3]:
            predicted, true_sense, rule, context = correct[0], correct[1], correct[2], correct[3]
            
            if rule == "default":
                print("Predicted: {}, was actually: {} \n Rule: {}, \n Sentence: {}"
                        .format(predicted, true_sense, "Used majority baseline", " ".join(context)))
            else:
                print("Predicted: {}, was actually: {} \n Rule: {}, log-likelihood: {:.3f} \n Sentence: {}"
                        .format(predicted, true_sense, rule[0], rule[1], " ".join(context)))
        
    
    def predict(self, context):
        """
        Predict word sense based on the new context given
        """
        
        for rule in self.decisionList:
            if self.check_rule(context[1], context[2], rule[0]):
                # If log-likelihood of rule is a neg #, it is a *word
                if rule[1] < 0:
                    return (self.word_star, context[0], rule, context[1])
                # If log-likelihood of rule is a pos #, it is a word
                elif rule[1] > 0:
                    return (self.word, context[0], rule, context[1])
        
        # If log == 0, resort to majority label
        return (self.majority_label, context[0], "default", context[1])
    
    def check_rule(self, context, tags, rule):
        
        r_scope, r_type, r_context = rule.split("_")
        r_scope = int(r_scope)
        
        if r_type == "tag":
            return self.get_k_tag(r_scope, context, tags) == r_context
        elif r_type == "word":
            return self.get_k_word(r_scope, context) == r_context
        elif r_type == "window":
            # Check all possible words within the window
            # to see if it matches the context 
            k_tmp = r_scope
            while k_tmp > 0:
                if self.get_k_word(k_tmp, context) == r_context:
                    return True
                if self.get_k_word(-1*k_tmp, context) == r_context:
                    return True 
                k_tmp -= 1
            return False 
        else:
            return False
    
    
def main():
    
    # Instantiate the parser 
    parser = argparse.ArgumentParser(description='Optional app description')
    
    # Take in train file
    parser.add_argument('-t', help='trainfile')
    
    # Take in test file
    parser.add_argument("-s", help="testfile")
    
    # Store values
    args = parser.parse_args()
    
    """
    training and test set result 
    """
    train = open(args.t, "r")
    test = open(args.s, "r")
    dl = DecisionList(train) 
    
    dl.fit()
    dl.evaluate(test_file=test)
    dl.print_eval()        
        
main()