import json
from utils.helper import *
from transformers import AlbertTokenizer, AutoTokenizer, AutoModelForPreTraining
from torch.utils.data import Dataset, DataLoader
import random


class dataprocess(Dataset):
    def __init__(self, data, mode, embed_mode, max_seq_len, dataset):
        self.data = data
        self.mode = mode
        self.len = max_seq_len
        self.embed_mode = embed_mode
        if embed_mode == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
        elif embed_mode == "bert_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif embed_mode == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # Longest sentence (precalculated):
        #       - ADE: 97 (+2 for the special tokens)
        #       - CONLL04: 118 (+2 for the special tokens)
        #       - SCIERC: 101 (+2 for the special tokens)
        if dataset == 'ADE':
            self.max_sentence_length = 99
        elif dataset == 'CONLL04':
            self.max_sentence_length = 120

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'pretrained_frozen_emb':
            input_embeddings = torch.FloatTensor(self.data[idx][3])

            ner_labels = self.data[idx][1]
            rc_labels = self.data[idx][2]

            ner_labels = self.ner_label_transform_1(ner_labels)
            rc_labels = self.rc_label_transform_1(rc_labels)

            # +2: '[CLS]' & '[SEP]' tokens
            mask_len = len(self.data[idx][0]) + 2

            return (input_embeddings, ner_labels, rc_labels, mask_len)
        elif self.mode == 'e2e_training':
            words = self.data[idx][0]
            ner_labels = self.data[idx][1]
            rc_labels = self.data[idx][2]

            # For ADE, CONLL04, SciERC, with maximum sentence length = 128 (parameter), truncation
            # will not be executed because the maximum sentence length is 118 in these datasets.
            if len(words) > self.len:
                words, ner_labels, rc_labels = self.truncate(self.len, words, ner_labels, rc_labels)

            sent_str = ' '.join(words)
            bert_words = self.tokenizer.tokenize(sent_str)
            bert_len = len(bert_words) + 2
            # bert_len = original sentence + [CLS] and [SEP]

            word_to_bep = self.map_origin_word_to_bert(words)
            ner_labels = self.ner_label_transform_2(ner_labels, word_to_bep)
            rc_labels = self.rc_label_transform_2(rc_labels, word_to_bep)

            return (words, ner_labels, rc_labels, bert_len)
        elif self.mode == 'e2e_training_word_level':
            if self.embed_mode == 'characterBERT':
                tokens_low = [t.lower() for t in self.data[idx][0]]
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

                ner_labels = self.data[idx][1]
                rc_labels = self.data[idx][2]
                ner_labels = self.ner_label_transform_1(ner_labels)
                rc_labels = self.rc_label_transform_1(rc_labels)

                return (tokens_final, ner_labels, rc_labels, attention_masks, mask_len)
            elif self.embed_mode.split('_')[0] == 'canine':
                words = self.data[idx][0]
                word_ranges = []
                for i, w in enumerate(words):
                    if i == 0:
                        word_ranges.append(list(range(1, 1 + len(w))))
                    else:
                        word_ranges.append(list(range(word_ranges[i - 1][-1] + 2, word_ranges[i - 1][-1] + 2 + len(w))))

                sent_str = ' '.join(words)
                ner_labels = self.data[idx][1]
                rc_labels = self.data[idx][2]
                ner_labels = self.ner_label_transform_1(ner_labels)
                rc_labels = self.rc_label_transform_1(rc_labels)

                # +2: '[CLS]' & '[SEP]' tokens
                mask_len = len(self.data[idx][0]) + 2

                return (sent_str, ner_labels, rc_labels, word_ranges, mask_len)
            else:
                words = self.data[idx][0]
                ner_labels = self.data[idx][1]
                rc_labels = self.data[idx][2]

                if len(words) > self.len:
                    words, ner_labels, rc_labels = self.truncate(self.len, words, ner_labels, rc_labels)

                mask_len = len(self.data[idx][0]) + 2

                word_to_bep = self.map_origin_word_to_bert(words)
                ner_labels = self.ner_label_transform_1(ner_labels)
                rc_labels = self.rc_label_transform_1(rc_labels)

                return (words, ner_labels, rc_labels, mask_len, word_to_bep)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform_1(self, ner_label):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = ner_label[i] + 1
            end = ner_label[i + 1] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform_1(self, rc_label):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = rc_label[i] + 1
            e2 = rc_label[i + 1] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels

    def ner_label_transform_2(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform_2(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels

    def truncate(self, max_seq_len, words, ner_labels, rc_labels):
        truncated_words = words[:max_seq_len]
        truncated_ner_labels = []
        truncated_rc_labels = []
        for i in range(0, len(ner_labels), 3):
            if ner_labels[i] < max_seq_len and ner_labels[i + 1] < max_seq_len:
                truncated_ner_labels += [ner_labels[i], ner_labels[i + 1], ner_labels[i + 2]]

        for i in range(0, len(rc_labels), 3):
            if rc_labels[i] < max_seq_len and rc_labels[i + 1] < max_seq_len:
                truncated_rc_labels += [rc_labels[i], rc_labels[i + 1], rc_labels[i + 2]]

        return truncated_words, truncated_ner_labels, truncated_rc_labels


def data_preprocess_1(data, dataset):
    processed = []
    for dic in data:
        text = dic['tokens']
        ner_labels = []
        rc_labels = []
        entity = dic['entities']
        relation = dic['relations']
        emb = dic['trained - embeddings']

        for en in entity:
            ner_labels += [en['start'], en['end'] - 1, en['type']]

        for re in relation:
            subj_idx = re['head']
            obj_idx = re['tail']
            subj = entity[subj_idx]
            obj = entity[obj_idx]
            rc_labels += [subj['start'], obj['start'], re['type']]

        overlap_pattern = False
        if dataset == "ADE":
            for i in range(0, len(ner_labels), 3):
                for j in range(i + 3, len(ner_labels), 3):
                    if is_overlap([ner_labels[i], ner_labels[i + 1]], [ner_labels[j], ner_labels[j + 1]]):
                        overlap_pattern = True
                        break
        if overlap_pattern == True:
            continue

        processed += [(text, ner_labels, rc_labels, emb)]
    return processed


def data_preprocess_2(data, dataset):
    processed = []
    for dic in data:
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
        if dataset == "ADE":
            for i in range(0, len(ner_labels), 3):
                for j in range(i + 3, len(ner_labels), 3):
                    if is_overlap([ner_labels[i], ner_labels[i + 1]], [ner_labels[j], ner_labels[j + 1]]):
                        overlap_pattern = True
                        break
        if overlap_pattern == True:
            continue

        processed += [(text, ner_labels, rc_labels)]
    return processed


def dataloader(args, ner2idx, rel2idx):
    if args.mode == 'pretrained_frozen_emb':
        if args.embed_mode == 'characterBERT':
            path = args.offline_emb_path + 'data_' + args.sub_mode + '_' + args.embed_mode + '/' + args.data
        else:
            path = args.offline_emb_path + 'data_' + args.sub_mode + '_' + args.embed_mode + '_' + args.word_pieces_aggregation + '/' + args.data
    elif args.mode == 'e2e_training':
        path = "data/" + args.data
    elif args.mode == 'e2e_training_word_level':
        path = "data/" + args.data

    if args.mode == 'e2e_training' or args.mode == 'e2e_training_word_level':
        if args.data == "ADE":
            train_raw_data = json_load(path, "raw/ade_split_" + str(args.split_id) + "_train.json")
            test_data = json_load(path, "raw/ade_split_" + str(args.split_id) + "_test.json")
            random.shuffle(train_raw_data)
            split = int(0.15 * len(train_raw_data))
            train_data = train_raw_data[split:]
            dev_data = train_raw_data[:split]
        elif args.data == 'CONLL04':
            train_data = json_load(path, 'train_triples.json')
            test_data = json_load(path, 'test_triples.json')
            dev_data = json_load(path, 'dev_triples.json')
    elif args.mode == 'pretrained_frozen_emb':
        if args.data == 'ADE':
            train_raw_data = json_load(path, "train_split_" + str(args.split_id) + ".json")
            test_data = json_load(path, "test_split_" + str(args.split_id) + ".json")
            random.shuffle(train_raw_data)
            split = int(0.15 * len(train_raw_data))
            train_data = train_raw_data[split:]
            dev_data = train_raw_data[:split]
        elif args.data == 'CONLL04':
            train_data = json_load(path, 'train_triples.json')
            test_data = json_load(path, 'test_triples.json')
            dev_data = json_load(path, 'dev_triples.json')

    if args.mode == 'pretrained_frozen_emb':
        train_data = data_preprocess_1(train_data, args.data)
        test_data = data_preprocess_1(test_data, args.data)
        dev_data = data_preprocess_1(dev_data, args.data)
    elif args.mode == 'e2e_training' or args.mode == 'e2e_training_word_level':
        train_data = data_preprocess_2(train_data, args.data)
        test_data = data_preprocess_2(test_data, args.data)
        dev_data = data_preprocess_2(dev_data, args.data)

    train_dataset = dataprocess(train_data, args.mode, args.embed_mode, args.max_seq_len, args.data)
    test_dataset = dataprocess(test_data, args.mode, args.embed_mode, args.max_seq_len, args.data)
    dev_dataset = dataprocess(dev_data, args.mode, args.embed_mode, args.max_seq_len, args.data)

    if args.mode == 'pretrained_frozen_emb':
        collate_fn = collater_1(ner2idx, rel2idx, args.data)
    elif args.mode == 'e2e_training':
        collate_fn = collater_2(ner2idx, rel2idx)
    elif args.mode == 'e2e_training_word_level':
        if args.embed_mode == 'characterBERT':
            collate_fn = collater_3(ner2idx, rel2idx, args.data)
        elif args.embed_mode.split('_')[0] == 'canine':
            collate_fn = collater_5(ner2idx, rel2idx, args.data)
        else:
            collate_fn = collater_4(ner2idx, rel2idx, args.data)

    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                             collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                            collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                           collate_fn=collate_fn)

    return train_batch, test_batch, dev_batch