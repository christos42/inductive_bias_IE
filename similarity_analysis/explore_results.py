import os
import json
import numpy as np
import argparse

class Search:
    def __init__(self, folder_path, filename):
        self.folder_path = folder_path
        self.filename = filename
        self.type = filename.split('_')[0]
        self.files_to_check = self.get_files()
        if self.type == 'ne':
            self.scores_per_model = self.get_scores_ne()
            self.mean_scores_per_model = self.get_mean_scores_ne()
        elif self.type == 're':
            self.scores_per_model = self.get_scores_re()
            self.mean_scores_per_model = self.get_mean_scores_re()

    def get_files(self):
        files_paths = {}
        for root, dirs, files in os.walk(self.folder_path, topdown=False):
            for name in files:
                if name == self.filename:
                    candidate_key = '/'.join(root.split('/')[:-1])
                    if candidate_key not in files_paths.keys():
                        files_paths[candidate_key] = []
                    files_paths[candidate_key].append(os.path.join(root, name))

        return files_paths

    def get_scores_ne(self):
        scores = {}
        for k in self.files_to_check:
            scores[k] = {}
            for f in self.files_to_check[k]:
                similarities = self.load_json(f)
                for k_s in similarities:
                    if k_s not in scores[k].keys():
                        scores[k][k_s] = {'start': [],
                                          'end': [],
                                          'joint': []}
                    scores[k][k_s]['start'].append(similarities[k_s]['start'])
                    scores[k][k_s]['end'].append(similarities[k_s]['end'])
                    scores[k][k_s]['joint'].append(similarities[k_s]['joint'])

        return scores

    def get_scores_re(self):
        scores = {}
        for k in self.files_to_check:
            scores[k] = {}
            for f in self.files_to_check[k]:
                similarities = self.load_json(f)
                for k_s in similarities:
                    if k_s not in scores[k].keys():
                        scores[k][k_s] = []

                    scores[k][k_s].append(similarities[k_s])

        return scores

    def get_mean_scores_ne(self):
        mean_scores = {}
        for k in self.scores_per_model:
            mean_scores[k] = {}
            for k_s in self.scores_per_model[k]:
                if k_s not in mean_scores[k].keys():
                    mean_scores[k][k_s] = {'start': np.mean(self.scores_per_model[k][k_s]['start']),
                                           'end': np.mean(self.scores_per_model[k][k_s]['end']),
                                           'joint': np.mean(self.scores_per_model[k][k_s]['joint'])}

        return mean_scores

    def get_mean_scores_re(self):
        mean_scores = {}
        for k in self.scores_per_model:
            mean_scores[k] = {}
            for k_s in self.scores_per_model[k]:
                if k_s not in mean_scores[k].keys():
                    mean_scores[k][k_s] = np.mean(self.scores_per_model[k][k_s])

        return mean_scores

    def load_json(self, path_file):
        with open(path_file) as json_file:
            data = json.load(json_file)

        return data

    def print_results(self):
        if self.type == 'ne':
            for k in self.mean_scores_per_model:
                print(k)
                for k_s in self.mean_scores_per_model[k]:
                    print(k_s)
                    for k_s_s in self.mean_scores_per_model[k][k_s]:
                        print('{}: {}' .format(k_s_s, self.mean_scores_per_model[k][k_s][k_s_s]))
                    print('__')
                print('################')
        elif self.type == 're':
            for k in self.mean_scores_per_model:
                print(k)
                for k_s in self.mean_scores_per_model[k]:
                    print('{}: {}' .format(k_s, self.mean_scores_per_model[k][k_s]))
                print('################')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default=None, type=str, required=False,
                        help="the filename of the file with the similarity calculations")

    args = parser.parse_args()
    search = Search('results/', args.filename)
    search.print_results()