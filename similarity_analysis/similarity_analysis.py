import sys
import argparse
import torch
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel, AutoModelForPreTraining
from transformers import CanineTokenizer, CanineModel   # Comment in when Environment 1 is used.
from representation_extraction import RepresentationExtraction

### Comment in when Environment 2 is used. (the next 3 lines)
#sys.path.append('../character-bert/')
#from utils.character_cnn import CharacterIndexer
#from modeling.character_bert import CharacterBertModel

### Comment in when Environment 2 is used. (the next 3 lines)
sys.path.append('../utils/')
from helper import *

class SimilarityStudy:
    def __init__(self, representations_ne, representations_re, output_dir):
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.representations_ne = representations_ne
        self.representations_re = representations_re
        self.representations_per_type_ne = self.listing_representations_ne()
        self.representations_per_type_re = self.listing_representations_re()
        self.similarities_ne = self.find_similarities_ne()
        self.similarities_re = self.find_similarities_re()
        self.output_dir = output_dir

    def print_similarities(self):
        print('ENTITIES:')
        for k in self.similarities_ne:
            print('Entity: {}' .format(k))
            print('Start-words: {:.4f}' .format(self.similarities_ne[k]['start']))
            print('End-words: {:.4f}' .format(self.similarities_ne[k]['end']))
            print('Overall: {:.4f}' .format(self.similarities_ne[k]['joint']))
            print('####################')
        print('--------------------------')
        print('--------------------------')
        print('RELATIONS:')
        for k in self.similarities_re:
            print('Relation: {}' .format(k))
            print('Similarity score: {:.4f}' .format(self.similarities_re[k]))

    def find_similarities_ne(self):
        sim_dict = {}
        for k in self.representations_per_type_ne:
            if k not in list(sim_dict.keys()):
                sim_dict[k] = {}
            sim_dict[k]['start'] = self.calc_similarities_ne(self.representations_per_type_ne[k]['start'])
            sim_dict[k]['end'] = self.calc_similarities_ne(self.representations_per_type_ne[k]['end'])
            sim_dict[k]['joint'] = self.calc_similarities_ne(self.representations_per_type_ne[k]['start'],
                                                             self.representations_per_type_ne[k]['end'])
        return sim_dict

    def find_similarities_re(self):
        sim_dict = {}
        for k in self.representations_per_type_re:
            sim_dict[k] = self.calc_similarities_re(self.representations_per_type_re[k])

        return sim_dict

    def calc_similarities_re(self, rep):
        similarities = []
        for r in rep:
            similarities.append(self.cos(torch.FloatTensor(r[0]), torch.FloatTensor(r[1])).item())

        return np.mean(similarities)

    def calc_similarities_ne(self, rep1, rep2 = []):
        similarities = []
        if len(rep2) > 1:
            rep = rep1 + rep2
        else:
            rep = rep1.copy()
        for i, r1 in enumerate(rep):
            for j, r2 in enumerate(rep):
                if i != j:
                    similarities.append(self.cos(torch.FloatTensor(r1), torch.FloatTensor(r2)).item())

        return np.mean(similarities)

    def listing_representations_ne(self):
        rep_dict = {}
        for rep in self.representations_ne:
            if rep[1] not in list(rep_dict.keys()):
                rep_dict[rep[1]] = {'start': [],
                                    'end': []}
            rep_dict[rep[1]][rep[2]].append(rep[0])

        return rep_dict

    def listing_representations_re(self):
        rep_dict = {}
        for rep in self.representations_re:
            if rep[2] not in list(rep_dict.keys()):
                rep_dict[rep[2]] = []

            rep_dict[rep[2]].append([rep[0], rep[1]])

        return rep_dict

    def save_similarities(self):
        json_save(self.output_dir + 'ne_similarities.json', self.similarities_ne)
        json_save(self.output_dir + 're_similarities.json', self.similarities_re)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="the name of the dataset: ADE or CONLL04")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="the path of the data")
    parser.add_argument("--data_name", default=None, type=str, required=True,
                        help="the name of the data file")
    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="the embedding mode/LM selection")
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="token_level or word_level")
    parser.add_argument("--aggregation", default='avg', type=str, required=False,
                        help="aggregation mode: avg or sum")
    parser.add_argument("--path_trained_model", default=None, type=str, required=True,
                        help="the path to the trained model")
    args = parser.parse_args()
    
    if args.dataset == 'ADE':
        set_name = args.data_name.split('.')[0].split('_')[-1]
        run_id = ''
    elif args.dataset == 'CONLL04':
        set_name = args.data_name.split('.')[0].split('_')[0]
        run_id = '_' + args.path_trained_model.split('/')[8].split('_')[1]
    if args.mode == 'word_level':
        output_dir = "results/" + args.dataset + '/' + args.embed_mode + '/' + args.mode + '/' + args.aggregation + '/' + set_name + '/' + args.path_trained_model.split('/')[-1][:-3] + run_id + '/'
    else:
        output_dir = "results/" + args.dataset + '/' + args.embed_mode + '/' + args.mode + '/' + set_name + '/' + args.path_trained_model.split('/')[-1][:-3] + run_id + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extraction = RepresentationExtraction(args.dataset,
                                          args.data_path,
                                          args.data_name,
                                          args.embed_mode,
                                          args.mode,
                                          args.aggregation,
                                          args.path_trained_model,
                                          device)

    rep_ne, rep_re = extraction.extract_representations()
    sim = SimilarityStudy(rep_ne, rep_re, output_dir)
    sim.save_similarities()