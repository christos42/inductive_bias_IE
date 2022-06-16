import configparser
import json
import numpy as np
import random
import torch
import os
import sys
from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
from transformers import CanineTokenizer, CanineModel
import argparse

sys.path.append('../character-bert/')
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel


class Embedding_Extraction:
    def __init__(self, args, data, device, path_out):
        self.args = args
        self.data = data
        self.device = device

        self.path_out = path_out
        # Create the output directory if it doesn't exist.
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)

        if self.args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.bert = AutoModel.from_pretrained("bert-base-cased").to(self.device)
        elif self.args.embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            self.bert = AlbertModel.from_pretrained("albert-xxlarge-v1").to(self.device)
        elif self.args.embed_mode == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        elif args.embed_mode == 'canine_c':
            self.tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
            self.bert = CanineModel.from_pretrained('google/canine-c').to(self.device)
        elif args.embed_mode == 'canine_s':
            self.tokenizer = CanineTokenizer.from_pretrained('google/canine-s')
            self.bert = CanineModel.from_pretrained('google/canine-s').to(self.device)
        elif self.args.embed_mode == 'characterBERT':
            if self.args.dataset == 'ADE':
                self.bert = CharacterBertModel.from_pretrained('../character-bert/pretrained-models/medical_character_bert/').to(self.device)
            else:
                self.bert = CharacterBertModel.from_pretrained('../character-bert/pretrained-models/general_character_bert/').to(self.device)
        else:
            print('Error: invalid pretrained model')

        # Longest sentence (precalculated):
        #       - ADE: 97 (+2 for the special tokens)
        #       - CONLL04: 118 (+2 for the special tokens)
        #       - SCIERC: 101 (+2 for the special tokens)
        if self.args.dataset == 'ADE':
            self.max_sentence_length = 99
        elif self.args.dataset == 'CONLL04':
            self.max_sentence_length = 120
        elif self.args.dataset == 'SCIERC':
            self.max_sentence_length = 103

    def extract_embeddings(self):
        if self.args.mode == 'word_level':
            if self.args.embed_mode == 'characterBERT':
                self.extract_embeddings_word_level_characterBERT()
            elif self.args.embed_mode.split('_')[0] == 'canine':
                self.extract_embeddings_word_level_canine()
            else:
                self.extract_embeddings_word_level_al_bert()
        elif self.args.mode == 'CLDR_CLNER':
            self.extract_embeddings_CLDR_CLNER()

    def extract_embeddings_word_level_al_bert(self):
        # Prepare for the extraction of the pre-trained Bert/Albert embeddings
        all_words = self.return_text()
        all_words_tokenized = self.tokenizer(all_words, return_tensors="pt", padding='longest', is_split_into_words=True)
        # Find mapping with the word-pieces
        word_to_bert = []
        for words in all_words:
            word_to_bert.append(self.map_origin_word_to_bert(words))

        final_data = []
        counter = 0
        for i, sent in enumerate(self.data):
            # Add '[CLS]' and '[SEP]' tokens
            tokens_low = [t.lower() for t in sent['tokens']]

            # Extraction from tuned CharacterBert
            tokens = ['[CLS]', *tokens_low, '[SEP]']

            # Find how many 0s should be added
            to_be_padded = self.max_sentence_length - len(tokens)
            zero_sec = [0] * to_be_padded
            one_sec = [1] * len(tokens)

            # Masking
            attention_masks = [one_sec + zero_sec]
            attention_masks = torch.LongTensor(attention_masks).to(self.device)
            #########################################################################################
            # Extract Bert/Albert embeddings
            tmp_input_ids = all_words_tokenized['input_ids'][i].unsqueeze(0).to(self.device)
            tmp_type_ids = all_words_tokenized['token_type_ids'][i].unsqueeze(0).to(self.device)
            tmp_attention_mask = all_words_tokenized['attention_mask'][i].unsqueeze(0).to(self.device)
            x_sin = {'input_ids': tmp_input_ids,
                     'token_type_ids': tmp_type_ids,
                     'attention_mask': tmp_attention_mask}
            with torch.no_grad():
                emb = self.bert(**x_sin)[0]
            # Add the embeddings for '[CLS]' token
            tmp_extracted_emb = [emb[0][0]]
            for w_k in word_to_bert[i].keys():
                start = word_to_bert[i][w_k][0] + 1
                end = word_to_bert[i][w_k][1] + 1
                emb_sel = emb[0][start:end + 1]
                if self.args.word_pieces_aggregation == 'avg':
                    tmp_extracted_emb.append(torch.mean(emb_sel, dim=0))
                elif self.args.word_pieces_aggregation == 'sum':
                    tmp_extracted_emb.append(torch.sum(emb_sel, dim=0))
            # Add the embeddings for '[SEP]' token
            tmp_extracted_emb.append(emb[0][end + 1])
            # Add the padded embeddings to have the same length for every sentence.
            padded_emb = emb[0][end + 2:self.max_sentence_length + end + 2 - len(tmp_extracted_emb)]
            for p_e in padded_emb:
                tmp_extracted_emb.append(p_e)

            # Remember: 99 is the length of the larger sentence (hard coded)
            if len(tmp_extracted_emb) != self.max_sentence_length:
                for j in range(self.max_sentence_length-len(tmp_extracted_emb)):
                    # Just add more times the last padded embedding vector
                    tmp_extracted_emb.append(p_e)

            merged_embeddings = []
            for emb_pretrained in tmp_extracted_emb:
                merged_embeddings.append(emb_pretrained.to(self.device))

            merged_embeddings_tensor = torch.stack(merged_embeddings, 0)
            merged_embeddings_tensor.squeeze_()

            dict_out = {'tokens': sent['tokens'],
                        'entities': sent['entities'],
                        'relations': sent['relations'],
                        'trained - embeddings': merged_embeddings_tensor.tolist(),
                        'attention mask': attention_masks[0].tolist()}

            final_data.append(dict_out)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed' .format(counter))

        self.save_final_data(final_data)

    def extract_embeddings_word_level_canine(self):
        final_data = []
        counter = 0
        for i, sent in enumerate(self.data):
            word_ranges = []
            for i, w in enumerate(sent['tokens']):
                if i == 0:
                    word_ranges.append(list(range(1, 1 + len(w))))
                else:
                    word_ranges.append(list(range(word_ranges[i - 1][-1] + 2, word_ranges[i - 1][-1] + 2 + len(w))))

            sent_str = ' '.join(sent['tokens'])
            x = self.tokenizer(sent_str, padding="longest",
                               return_tensors="pt").to(device)
            with torch.no_grad():
                emb = self.bert(**x)[0]
            # Add the embeddings for '[CLS]' token
            tmp_extracted_emb = [emb[0][0]]
            for i, w_r in enumerate(word_ranges):
                emb_sel = emb[0][w_r]
                if self.args.word_pieces_aggregation == 'avg':
                    tmp_extracted_emb.append(torch.mean(emb_sel, dim=0))
                elif self.args.word_pieces_aggregation == 'sum':
                    tmp_extracted_emb.append(torch.sum(emb_sel, dim=0))

            # Add the embeddings for '[SEP]' token
            tmp_extracted_emb.append(emb[0][w_r[-1] + 1])
            # Add artificially padded tokens
            tmp_extracted_emb.extend(torch.zeros(self.max_sentence_length - len(tmp_extracted_emb), 768))

            merged_embeddings = []
            for emb_pretrained in tmp_extracted_emb:
                merged_embeddings.append(emb_pretrained.to(self.device))

            merged_embeddings_tensor = torch.stack(merged_embeddings, 0)
            merged_embeddings_tensor.squeeze_()

            # Find how many 0s should be added
            # (len(sent['tokens']) + 2): +2 to include [CLS] and [SEP] tokens
            to_be_padded = self.max_sentence_length - (len(sent['tokens']) + 2)
            zero_sec = [0] * to_be_padded
            one_sec = [1] * (len(sent['tokens']) + 2)

            # Masking
            attention_masks = [one_sec + zero_sec]
            attention_masks = torch.LongTensor(attention_masks).to(self.device)

            dict_out = {'tokens': sent['tokens'],
                        'entities': sent['entities'],
                        'relations': sent['relations'],
                        'trained - embeddings': merged_embeddings_tensor.tolist(),
                        'attention mask': attention_masks[0].tolist()}

            final_data.append(dict_out)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed' .format(counter))

        self.save_final_data(final_data)

    def extract_embeddings_word_level_characterBERT(self):
        final_data = []
        counter = 0
        for sent in self.data:
            # Add '[CLS]' and '[SEP]' tokens
            tokens_low = [t.lower() for t in sent['tokens']]
            tokens = ['[CLS]', *tokens_low, '[SEP]']

            # Find how many 0s should be added
            to_be_padded = self.max_sentence_length - len(tokens)
            zero_sec = [0] * to_be_padded
            one_sec = [1] * len(tokens)

            # Padding
            padded_tokens = tokens + ['[PAD]'] * to_be_padded

            # Masking
            attention_masks = [one_sec + zero_sec]
            attention_masks = torch.LongTensor(attention_masks).to(self.device)

            # Convert token sequence into character indices
            indexer = CharacterIndexer()
            batch = [padded_tokens]  # This is a batch with a single token sequence
            batch_ids = indexer.as_padded_tensor(batch).to(self.device)

            with torch.no_grad():
                out_text_encoder = self.bert(batch_ids, attention_masks)[0][0]

            dict_out = {'tokens': sent['tokens'],
                        'entities': sent['entities'],
                        'relations': sent['relations'],
                        'trained - embeddings': out_text_encoder.tolist(),
                        'attention mask': attention_masks[0].tolist()}

            final_data.append(dict_out)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed' .format(counter))

        self.save_final_data(final_data)

    def save_final_data(self, final_data):
        if self.args.split_id == '-1':
            with open(self.path_out + self.args.set + '_triples.json', 'w') as fp:
                json.dump(final_data, fp)
        else:
            with open(self.path_out + self.args.set + '_split_' + self.args.split_id + '.json', 'w') as fp:
                json.dump(final_data, fp)

    def return_text(self):
        all_words = []
        for rec in self.data:
            t_l = [t.lower() for t in rec['tokens']]
            all_words.append(t_l)

        return all_words

    def map_origin_word_to_bert(self, words):
        bert_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bert_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bert_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="possible datasets: ADE, CONLL04")
    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="BERT, ALBERT, CharacterBERT, CANINE-c, CANINE-s pretrained embedding // accepted values: bert_cased, albert, characterBERT, canine_c, canine_s")
    parser.add_argument("--set", default=None, type=str, required=True,
                        help="accepted values: - ADE: train, test - CONLL04: train, dev, test")
    parser.add_argument("--split_id", default=None, type=str, required=True,
                        help="ADE: 10 fold split, accepted values: 0-9 // CONLL04: No splits, accepted value: -1")
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="Modes: word_level")
    parser.add_argument("--word_pieces_aggregation", default='avg', type=str,
                        help="aggregation strategy in case of word-pieces split, accepted values: avg (average), sum (summation)")
    parser.add_argument("--path_out", default=None, type=str, required=True,
                        help="Output path for the extracted embeddings")

    args = parser.parse_args()

    print('Processing...')
    if args.dataset == 'ADE':
        data_input_path = 'data/ADE/raw/ade_split_' + args.split_id + '_' + args.set + '.json'
    elif args.dataset == 'CONLL04':
        data_input_path = 'data/CONLL04/' + args.set + '_triples.json'
    else:
        print('Wrong dataset given.')

    with open(data_input_path) as json_file:
        data = json.load(json_file)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    if args.embed_mode == 'characterBERT':
        path_out = args.path_out + 'data_' + args.mode + '_' + args.embed_mode + '/' + args.dataset + '/'
    else:
        path_out = args.path_out + 'data_' + args.mode + '_' + args.embed_mode + '_' + args.word_pieces_aggregation + '/' + args.dataset + '/'

    extract_embeddings = Embedding_Extraction(args, data, device, path_out)

    extract_embeddings.extract_embeddings()