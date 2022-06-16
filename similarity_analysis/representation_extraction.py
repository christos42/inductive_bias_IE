from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel, AutoModelForPreTraining
from transformers import CanineTokenizer, CanineModel
import sys
sys.path.append('../utils/')
from helper import *


class RepresentationExtraction:
    def __init__(self, dataset, data_path, data_name, embed_mode, mode, aggregation, path_trained_model, device):
        # Longest sentence (precalculated):
        #       - ADE: 97 (+2 for the special tokens)
        #       - CONLL04: 118 (+2 for the special tokens)
        self.dataset = dataset
        if dataset == 'ADE':
            self.max_sentence_length = 99
        elif dataset == 'CONLL04':
            self.max_sentence_length = 120

        self.device = device
        self.data = json_load(data_path, data_name)
        self.mode = mode
        self.aggregation = aggregation

        self.embed_mode = embed_mode
        if embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            self.lm = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.lm = AutoModel.from_pretrained("bert-base-cased")
        elif embed_mode == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.lm = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif embed_mode == 'canine_c':
            self.tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
            self.lm = CanineModel.from_pretrained('google/canine-c')
        elif embed_mode == 'canine_s':
            self.tokenizer = CanineTokenizer.from_pretrained(
                'google/canine-s')  # Comment in when Environment 1 is used.
            self.lm = CanineModel.from_pretrained('google/canine-s')  # Comment in when Environment 1 is used.
        ### Comment in when Environment 2 is used. (the next 6 lines)
        elif embed_mode == 'characterBERT':
            self.indexer = CharacterIndexer()
            if dataset == 'ADE':
                self.lm = CharacterBertModel.from_pretrained('../character-bert/pretrained-models/medical_character_bert/')
            else:
                self.lm = CharacterBertModel.from_pretrained('../character-bert/pretrained-models/general_character_bert/')

        # Load the trained model
        self.trained_model = torch.load(path_trained_model)
        #print(self.trained_model.items())
        trained_dict = {'.'.join(k.split('.')[1:]): v for k, v in self.trained_model.items() if '.'.join(k.split('.')[1:]) in self.lm.state_dict().keys()}
        self.lm.load_state_dict(trained_dict)
        self.lm.to(device)
        self.lm.eval()

    def extract_representations(self):
        representations_ne = []
        representations_re = []
        processed_data = self.data_preprocess()
        for sent in processed_data:
            if self.mode == 'token_level':
                words, ner_labels, rc_labels = self.prepare_input_labels_1(sent)
                x = self.tokenizer(words, return_tensors="pt",
                                   is_split_into_words=True).to(self.device)
                x = self.lm(**x)[0][0]
                for en in ner_labels:
                    if en[0] != en[1]:
                        representations_ne.append((x[en[0]].tolist(), en[2], 'start'))
                        representations_ne.append((x[en[1]].tolist(), en[2], 'end'))
                    else:
                        representations_ne.append((x[en[0]].tolist(), en[2], 'start'))
                for rel in rc_labels:
                    representations_re.append((x[rel[0]].tolist(), x[rel[1]].tolist(), rel[2]))
            elif self.mode == 'word_level':
                if self.embed_mode == 'characterBERT':
                    words, ner_labels, rc_labels, attention_masks = self.prepare_input_labels_2(sent)
                    # Convert token sequence into character indices
                    batch_ids = self.indexer.as_padded_tensor(words).to(self.device)
                    attention_masks = attention_masks.to(self.device)

                    x = self.lm(batch_ids, attention_masks)[0][0]
                    for en in ner_labels:
                        if en[0] != en[1]:
                            representations_ne.append((x[en[0]].tolist(), en[2], 'start'))
                            representations_ne.append((x[en[1]].tolist(), en[2], 'end'))
                        else:
                            representations_ne.append((x[en[0]].tolist(), en[2], 'start'))
                    for rel in rc_labels:
                        representations_re.append((x[rel[0]].tolist(), x[rel[1]].tolist(), rel[2]))
                elif self.embed_mode.split('_')[0] == 'canine':
                    words, ner_labels, rc_labels, word_ranges = self.prepare_input_labels_3(sent)
                    x = self.tokenizer(words, return_tensors="pt",
                                       is_split_into_words=True).to(self.device)
                    x = self.lm(**x)[0][0]
                    # Append 'CLS' token representation
                    x_words = [x[0]]
                    for w_r in word_ranges:
                        emb_sel = x[w_r]
                        if self.aggregation == 'avg':
                            x_words.append(torch.mean(emb_sel, dim=0))
                        elif self.aggregation == 'sum':
                            x_words.append(torch.sum(emb_sel, dim=0))
                    for en in ner_labels:
                        if en[0] != en[1]:
                            representations_ne.append((x_words[en[0]].tolist(), en[2], 'start'))
                            representations_ne.append((x_words[en[1]].tolist(), en[2], 'end'))
                        else:
                            representations_ne.append((x_words[en[0]].tolist(), en[2], 'start'))
                    for rel in rc_labels:
                        representations_re.append((x_words[rel[0]].tolist(), x_words[rel[1]].tolist(), rel[2]))
                else:
                    words, ner_labels, rc_labels, word_to_bert = self.prepare_input_labels_4(sent)
                    x = self.tokenizer(words, return_tensors="pt",
                                       is_split_into_words=True).to(self.device)
                    x = self.lm(**x)[0][0]
                    # Append 'CLS' token representation
                    x_words = [x[0]]
                    for w_k in word_to_bert:
                        start = word_to_bert[w_k][0] + 1
                        end = word_to_bert[w_k][1] + 1
                        emb_sel = x[start:end + 1]
                        if self.aggregation == 'avg':
                            x_words.append(torch.mean(emb_sel, dim=0))
                        elif self.aggregation == 'sum':
                            x_words.append(torch.sum(emb_sel, dim=0))
                    for en in ner_labels:
                        if en[0] != en[1]:
                            representations_ne.append((x_words[en[0]].tolist(), en[2], 'start'))
                            representations_ne.append((x_words[en[1]].tolist(), en[2], 'end'))
                        else:
                            representations_ne.append((x_words[en[0]].tolist(), en[2], 'start'))
                    for rel in rc_labels:
                        representations_re.append((x_words[rel[0]].tolist(), x_words[rel[1]].tolist(), rel[2]))

        return representations_ne, representations_re

    def prepare_input_labels_4(self, sent):
        words = sent[0]
        ner_labels = sent[1]
        rc_labels = sent[2]

        #mask_len = len(sent[0]) + 2

        word_to_bert = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform_2(ner_labels)
        rc_labels = self.rc_label_transform_2(rc_labels)

        return words, ner_labels, rc_labels, word_to_bert
        #return (words, ner_labels, rc_labels, mask_len, word_to_bep)

    def prepare_input_labels_3(self, sent):
        words = sent[0]
        word_ranges = []
        for i, w in enumerate(words):
            if i == 0:
                word_ranges.append(list(range(1, 1 + len(w))))
            else:
                word_ranges.append(list(range(word_ranges[i - 1][-1] + 2, word_ranges[i - 1][-1] + 2 + len(w))))

        sent_str = ' '.join(words)
        ner_labels = sent[1]
        rc_labels = sent[2]
        ner_labels = self.ner_label_transform_2(ner_labels)
        rc_labels = self.rc_label_transform_2(rc_labels)

        # +2: '[CLS]' & '[SEP]' tokens
        #mask_len = len(sent[0]) + 2

        return sent_str, ner_labels, rc_labels, word_ranges
        #return (sent_str, ner_labels, rc_labels, word_ranges, mask_len)

    def prepare_input_labels_2(self, sent):
        tokens_low = [t.lower() for t in sent[0]]
        tokens_ready = ['[CLS]', *tokens_low, '[SEP]']

        mask_len = len(tokens_ready)

        # Find how many 0s should be added
        to_be_padded = self.max_sentence_length - len(tokens_ready)
        zero_sec = [0] * to_be_padded
        one_sec = [1] * len(tokens_ready)

        # Padding
        tokens_final = tokens_ready + ['[PAD]'] * to_be_padded

        # Masking
        attention_masks = [one_sec + zero_sec]
        attention_masks = torch.LongTensor(attention_masks)

        ner_labels = sent[1]
        rc_labels = sent[2]
        ner_labels = self.ner_label_transform_2(ner_labels)
        rc_labels = self.rc_label_transform_2(rc_labels)

        return tokens_final, ner_labels, rc_labels, attention_masks
        #return (tokens_final, ner_labels, rc_labels, attention_masks, mask_len)

    def prepare_input_labels_1(self, sent):
        words = sent[0]
        ner_labels = sent[1]
        rc_labels = sent[2]

        sent_str = ' '.join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        word_to_bert = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform_1(ner_labels, word_to_bert)
        rc_labels = self.rc_label_transform_1(rc_labels, word_to_bert)

        return words, ner_labels, rc_labels
        #return (words, ner_labels, rc_labels, bert_len)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform_1(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels.append([sta, end, ner_label[i + 2]])

        return new_ner_labels

    def rc_label_transform_1(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels.append([e1, e2, rc_label[i + 2]])

        return new_rc_labels

    def ner_label_transform_2(self, ner_label):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = ner_label[i] + 1
            end = ner_label[i + 1] + 1
            new_ner_labels.append([sta, end, ner_label[i + 2]])

        return new_ner_labels

    def rc_label_transform_2(self, rc_label):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = rc_label[i] + 1
            e2 = rc_label[i + 1] + 1
            new_rc_labels.append([e1, e2, rc_label[i + 2]])

        return new_rc_labels

    def data_preprocess(self):
        processed = []
        for dic in self.data:
            text = dic['tokens']
            ner_labels = []
            rc_labels = []
            entity = dic['entities']
            relation = dic['relations']

            for en in entity:
                ner_labels += [en['start'], en['end'] - 1, en['type']]

            for re in relation:
                subj_idx = re['head']
                obj_idx = re['tail']
                subj = entity[subj_idx]
                obj = entity[obj_idx]
                rc_labels += [subj['start'], obj['start'], re['type']]

            overlap_pattern = False
            if self.dataset == "ADE":
                for i in range(0, len(ner_labels), 3):
                    for j in range(i + 3, len(ner_labels), 3):
                        if is_overlap([ner_labels[i], ner_labels[i + 1]], [ner_labels[j], ner_labels[j + 1]]):
                            overlap_pattern = True
                            break
            if overlap_pattern == True:
                continue

            processed += [(text, ner_labels, rc_labels)]
        return processed