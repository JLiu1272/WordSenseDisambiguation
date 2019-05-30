"""
File: ps2_2_extexts.py
Author: Jennifer Liu
"""

#from stanfordcorenlp import StanfordCoreNLP
import json
from collections import defaultdict
import argparse 

# Spacy 
import spacy 

"""
SpaCy implementation of tokenization, Name Entity Recognition, 
Parts of Speech Tagging, Sentence Splitting 
"""
class SpaCy:
    def __init__(self):
        self.nlp = spacy.load('en')
    
    def pos(self, sentence):
        poses = []
        doc = self.nlp(sentence)
        
        for token in doc:
            poses.append((token.text, token.pos_))
        
        return poses
    
    def word_tokenize(self, sentence):
        doc = self.nlp(sentence)
        tokens = []
        
        for token in doc:
            tokens.append(token.text)
        
        return tokens

    def ner(self, sentence):
        doc = self.nlp(sentence)
        ners = []
        
        for ent in doc.ents:
            ners.append((ent.text, ent.label_))
        
        return ners 
    
    def lemma(self, sentence):
        lemmas = []
        doc = self.nlp(sentence)
        
        for token in doc:
            lemmas.append(token.lemma_)
            
        return lemmas    
        
    def sentence_splitting(self, sentence):
        sentences = []
        doc = self.nlp(sentence)
        
        for sent in doc.sents:
            sentences.append(sent.text)
        
        return sentences 


"""
StandfordNLP implementation of tokenization, Name Entity Recognition, 

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)
    
    
    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))
    
    def lemma(self, annotate):
        lemmas = []
        
        for i in range(len(annotate['sentences'])):    
            lemmas.extend([v for d in annotate['sentences'][i]['tokens'] for k,v in d.items() if k == 'lemma'])
        return lemmas    
        
    def sentence_splitting(self, annotate):
        sentences = []
        
        for i in range(len(annotate['sentences'])):
            sentence = [v for d in annotate['sentences'][i]['tokens'] for k,v in d.items() if k == 'originalText']
            sentences.append(" ".join(sentence))
            
        return sentences 
    
    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens
"""

class FileHandler:
    def __init__(self):
        pass
    
    def parse_file(self, filename):
        file = open(filename, "r")
        sentences = []
        
        for line in file:
            sentences.append(line)
        
        return sentences
    
    def show_stats(self, sentences, nlp):
        
        if nlp == "CoreNLP":
            #sNLP = StanfordNLP()
            pass
        elif nlp == "Spacy":
            sNLP = SpaCy()
        
        """
        Spacy and CoreNLP have slightly different
        implementation for lemmetization and sentence
        splitting. Therefore, there are if statements
        to verify which action to take
        """
        
        for text in sentences:
            print("Text: ")
            print(text)
            print()
            
            if nlp == "CoreNLP":
                parsed_dict = sNLP.annotate(text)
            
            print("Sentence Splitting: ")
            if nlp == "CoreNLP":
                print(sNLP.sentence_splitting(parsed_dict))
            else:
                print(sNLP.sentence_splitting(text))
            print()
            
            print("Lemmetization: ")
            if nlp == "CoreNLP": 
                print(sNLP.lemma(parsed_dict))
            else:
                print(sNLP.lemma(text))
            print()
            
            print("Tokenization: ")
            print(sNLP.word_tokenize(text))
            print()
            
            print("Part of Speech Tagging: ")
            print(sNLP.pos(text))
            print()
            
            print("Name Entity Recognition: ")
            print(sNLP.ner(text))
            print()


if __name__ == '__main__':
    
    # Instantiate the parser 
    parser = argparse.ArgumentParser(description='Optional app description')
    
    # Take in mysentences text
    parser.add_argument('sentences', help='A file containing the sentences')
    
    args = parser.parse_args()
    
    # Init file Handler object 
    fileHandler = FileHandler() 
    
    # Parse sentences
    sentences = fileHandler.parse_file(args.sentences)
    
    """
    Stanford Core NLP results 
    
    # show stats
    fileHandler.show_stats(sentences, "CoreNLP")
    """
    
    """
    SpaCy results 
    """
    # Show stats
    fileHandler.show_stats(sentences, "Spacy")
    
    
    