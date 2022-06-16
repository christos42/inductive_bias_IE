import json
import matplotlib.pyplot as plt
import argparse

class Hyphenation:
    def __init__(self, dataset, step, first_n):
        self.dataset = dataset
        self.step = step
        self.first_n = first_n
        self.data = self.load_data()
        if dataset == 'ADE':
            self.entities = {'Drug': [],
                             'Adverse-Effect': []}
        elif dataset == 'CONLL04':
            self.entities = {'Peop': [],
                             'Loc': [],
                             'Other': [],
                             'Org': []}
        self.find_entities()
        self.all_entities_flattened = self.flatten_entities()
        self.add_outside_words()
        self.unique_entities = self.find_unique_entities()
        self.flattened_unique_entities_words = self.flatten_unique_entities_words()
        self.hyphenated_unique_entities_subwords = self.find_hyphenated_unique_entities_subwords()
        self.hyphenated_unique_entities_subwords_frequency = self.find_hyphenated_unique_entities_subwords_frequency()
        self.sorted_hyphenated_unique_entities_subwords_frequency = self.sort_hyphenated_unique_entities_subwords_frequency()
        
        # Find the 50 most frequent subwords of the 'Outside' words.
        self.freq_50_subwords_outside = []
        for t in list(self.sorted_hyphenated_unique_entities_subwords_frequency['Outside'].items())[:50]:
            self.freq_50_subwords_outside.append(t[0])
        
    def load_data(self):
        if self.dataset == 'ADE':
            f = open('../data/ADE/raw/ade_split_0_test.json')
            data_test = json.load(f)

            f = open('../data/ADE/raw/ade_split_0_train.json')
            data_train = json.load(f)

            data_all = data_test + data_train 
        elif self.dataset == 'CONLL04':
            f = open('../data/CONLL04/dev_triples.json')
            data_dev = json.load(f)

            f = open('../data/CONLL04/test_triples.json')
            data_test = json.load(f)

            f = open('../data/CONLL04/train_triples.json')
            data_train = json.load(f)

            data_all = []
            data_all.extend(data_dev)
            data_all.extend(data_test)
            data_all.extend(data_train)
            
        return data_all
    
    def find_entities(self):
        for s in self.data:
            for en in s['entities']:
                self.entities[en['type']].append([t.lower() for t in s['tokens'][en['start']:en['end']]])
    
    def flatten_entities(self):
        all_entities_flattened = []
        for k in self.entities.keys():
            for en in self.entities[k]:
                for s_en in en:
                    all_entities_flattened.append(s_en)
                    
        return all_entities_flattened
    
    def add_outside_words(self):
        self.entities['Outside'] = []
        for s in self.data:
            for t in s['tokens']:
                if t.lower() not in self.all_entities_flattened:
                    self.entities['Outside'].append([t.lower()])
    
    def find_unique_entities(self): 
        unique_entities = {}
        for k in self.entities.keys():
            unique_entities[k] = []
            for en in self.entities[k]:
                if en not in unique_entities[k]:
                    unique_entities[k].append(en)
        
        return unique_entities
    
    def flatten_unique_entities_words(self):
        flattened_unique_entities_words = {}
        for k in self.unique_entities.keys():
            flattened_unique_entities_words[k] = []
            for en in self.unique_entities[k]:
                for s_en in en:
                    if s_en not in flattened_unique_entities_words[k]:
                        flattened_unique_entities_words[k].append(s_en)
        
        return flattened_unique_entities_words
    
    def find_hyphenated_unique_entities_subwords(self):
        hyphenated_unique_entities_subwords = {}
        for k in self.flattened_unique_entities_words.keys():
            hyphenated_unique_entities_subwords[k] = []
            for w in self.flattened_unique_entities_words[k]:
                hyphenated_unique_entities_subwords[k].append(self.find_subphrases(w))
        
        return hyphenated_unique_entities_subwords
    
    def find_subphrases(self, word):
        subphrases = []
        for l in range(len(word)):
            if l+self.step <= len(word):
                subphrases.append(word[l:l+self.step])
    
        return subphrases
    
    def find_hyphenated_unique_entities_subwords_frequency(self):
        hyphenated_unique_entities_subwords_frequency = {}
        for k in self.hyphenated_unique_entities_subwords.keys():
            hyphenated_unique_entities_subwords_frequency[k] = {}
            for hyph in self.hyphenated_unique_entities_subwords[k]:
                for h in hyph:
                    if h not in hyphenated_unique_entities_subwords_frequency[k].keys():
                        hyphenated_unique_entities_subwords_frequency[k][h] = 1
                    else:
                        hyphenated_unique_entities_subwords_frequency[k][h] += 1
        
        return hyphenated_unique_entities_subwords_frequency
    
    def sort_hyphenated_unique_entities_subwords_frequency(self):
        sorted_hyphenated_unique_entities_subwords_frequency = {}
        for k in self.hyphenated_unique_entities_subwords.keys():
            sorted_hyphenated_unique_entities_subwords_frequency[k] = dict(sorted(self.hyphenated_unique_entities_subwords_frequency[k].items(), key=lambda item: item[1], reverse=True))
        
        return sorted_hyphenated_unique_entities_subwords_frequency
    
    def get_sorted_subwords_frequency(self):
        return self.sorted_hyphenated_unique_entities_subwords_frequency
    
    def create_bar_plots(self):
        for k in self.sorted_hyphenated_unique_entities_subwords_frequency.keys():
            most_freq_subwords = list(self.sorted_hyphenated_unique_entities_subwords_frequency[k].items())[:self.first_n]
            subwords, subwords_freq = [], []
            for t in most_freq_subwords:
                subwords.append(t[0])
                subwords_freq.append(t[1])
            
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams["figure.figsize"] = (10, 6)
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
            plt.bar(subwords, subwords_freq)
            plt.xticks(rotation='vertical', fontsize=18)
            plt.title(self.dataset + ' dataset: ' + str(k) + ' entity', fontsize=24)
            plt.xlabel('Subwords', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.savefig(self.dataset + '_dataset_' + str(k) + '_entity_' + str(self.first_n) + '_most_freq_' + str(self.step) + '_subword_length.png', dpi=1300) 
            plt.savefig(self.dataset + '_dataset_' + str(k) + '_entity_' + str(self.first_n) + '_most_freq_' + str(self.step) + '_subword_length.pdf', dpi=1300) 
            plt.close()
            
    def create_filtered_bar_plots(self):
        for k in self.sorted_hyphenated_unique_entities_subwords_frequency.keys():
            if k == 'Outside':
                most_freq_subwords = list(self.sorted_hyphenated_unique_entities_subwords_frequency[k].items())[:self.first_n]
            else:
                most_freq_subwords = self.filter_freq_subwords_based_on_overlap(k)
            subwords, subwords_freq = [], []
            for t in most_freq_subwords:
                subwords.append(t[0])
                subwords_freq.append(t[1])
            
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams["figure.figsize"] = (10, 6)
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
            plt.bar(subwords, subwords_freq)
            plt.xticks(rotation='vertical', fontsize=18)
            plt.title(self.dataset + ' dataset: ' + str(k) + ' entity', fontsize=24)
            plt.xlabel('Subwords', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.savefig(self.dataset + '_dataset_' + str(k) + '_entity_' + str(self.first_n) + '_most_freq_' + str(self.step) + '_subword_length_filtered.png', dpi=1300) 
            plt.savefig(self.dataset + '_dataset_' + str(k) + '_entity_' + str(self.first_n) + '_most_freq_' + str(self.step) + '_subword_length_filtered.pdf', dpi=1300) 
            plt.close()
            
    def filter_freq_subwords_based_on_overlap(self, k):
        filtered_list = []
        c = 0
        for t in list(self.sorted_hyphenated_unique_entities_subwords_frequency[k].items()):
            if t[0] not in self.freq_50_subwords_outside:
                c += 1
                filtered_list.append(t)
            if c == self.first_n:
                break
        
        return filtered_list
        
    def find_overlap_of_freq_subwords(self):
        freq_subwords_outside = list(self.sorted_hyphenated_unique_entities_subwords_frequency['Outside'].items())[:self.first_n]
        freq_subwords_only_outside = []
        for t in freq_subwords_outside:
            freq_subwords_only_outside.append(t[0])

        for k in self.sorted_hyphenated_unique_entities_subwords_frequency.keys():
            if k == 'Outside':
                continue
            count_overlap = 0
            most_freq_subwords = list(self.sorted_hyphenated_unique_entities_subwords_frequency[k].items())[:self.first_n]
            for t in most_freq_subwords:
                if t[0] in freq_subwords_only_outside:
                    count_overlap += 1
            
            print('Entity {}: {} overlap' .format(k, count_overlap))
            
    def print_overlap_of_freq_subwords(self):
        freq_subwords_outside = list(self.sorted_hyphenated_unique_entities_subwords_frequency['Outside'].items())[:self.first_n]
        freq_subwords_only_outside = []
        for t in freq_subwords_outside:
            freq_subwords_only_outside.append(t[0])

        for k in self.sorted_hyphenated_unique_entities_subwords_frequency.keys():
            if k == 'Outside':
                continue
            print('Entity: {}' .format(k))
            most_freq_subwords = list(self.sorted_hyphenated_unique_entities_subwords_frequency[k].items())[:self.first_n]
            for t in most_freq_subwords:
                if t[0] in freq_subwords_only_outside:
                    print(t[0])
            print('################')
    
    def count_most_freq_words_with_threshold(self, threshold):
        for k in self.sorted_hyphenated_unique_entities_subwords_frequency.keys():
            c = 0
            if k == 'Outside':
                continue
            else:
                most_freq_subwords = self.filter_freq_subwords_based_on_overlap(k)
            for t in most_freq_subwords:
                if t[1] > threshold:
                    c += 1
            print('Entity {}: {} subwords' .format(k, c))


if __name__ == '__main__':
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="which dataset to use, ADE or CONLL04")
    parser.add_argument("--subword_length", default=None, type=int, required=True,
                        help="the length of the subwords")
    parser.add_argument("--first_n", default=None, type=int, required=True,
                        help="the first n most frequent words")
    parser.add_argument("--threshold", default=None, type=int, required=True,
                        help="the frequency threshold")
    args = parser.parse_args()

    hyph = Hyphenation(args.data, args.subword_length, args.first_n)

    print('Count the words with a frequency higher than a given threshold:')
    hyph.count_most_freq_words_with_threshold(args.threshold)
    
    # Create the plots
    hyph.create_filtered_bar_plots()
