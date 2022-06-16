# An Information Extraction Study: Take In Mind the Tokenization! 
Official code repository of "An Information Extraction Study: Take In Mind the Tokenization!" paper 
(PyTorch implementation).
For a description of the models and experiments, see our paper: (accepted at EMNLP 2022).

## Setup
### Requirements
We suggest to build two different environments. 
For the models that use CharacterBERT as Language Model we have the following requirements (environment 1):
 - Python 3.8+
 - PyTorch (tested with version 1.10.0) 
 - cudatoolkit (tested with version 10.2)
 - sentencepiece (tested with version 0.1.95)
 - transformers (tested with version 3.3.1)
 - tqdm (tested with version 4.62.3)
 - numpy (tested with version 1.21.2)

You can also find instructions on building the environment in the official <a target="_blank" href="https://github.com/helboukkouri/character-bert">repository</a> of CharacterBERT.  

For the rest of the models we have the following requirements (environment 2):
 - Python 3.8+
 - PyTorch (tested with version 1.10.0) 
 - cudatoolkit (tested with version 11.3.1)
 - sentencepiece (tested with version 0.1.95)
 - transformers (tested with version 4.11.3)
 - tqdm (tested with version 4.62.3)
 - numpy (tested with version 1.21.2)

 We use two different environments because there are some incompatibility issues with transformers library. CharacterBERT requires an older version while CANINE uses a newer version. 


### Execution steps
- Download the CharacterBERT model \[1\] using the official <a target="_blank" href="https://github.com/helboukkouri/character-bert">repository</a>. We are going to use the general and medical versions of CharacterBERT. Place the ```character-bert``` folder inside the repository folder. 
- Extract the pre-trained embeddings of the different Language Models by running the scripts under the ```/run_scripts/offline_embedding_extraction/``` folder. 
- Train the models in the baseline setup using the scripts under the ```/run_scripts/pretrained_frozen_emb/``` folder. 
- Train the models in the advanced setup (end-to-end training) using the scripts under the ```/run_scripts/e2e_training/``` and ```/run_scripts/e2e_training_word_level/``` folders. The ```*_word_level``` indicates the case where aggregation is used.
- Conduct the similarity analysis using the scripts under the ```/run_scripts/similarity_analysis/``` folder. Update the full path of the trained models in the running scripts accordingly.
- Use the ```explore_results.py``` script under the ```/similarity_analysis/``` folder to print and explore the aggregated results per model. (e.g. <i>python explore_results.py --filename 'ne_similarities.json'</i>)
- For the hyphenation analysis, execute the script under the corresponding folder (e.g. <i>python hyphenation_analysis.py --data 'ADE' --subword_length 4 --first_n 25 --threshold 20</i>)


## Comments
- The trained models are stored under the ```/save/``` folder.
- Change the paths in the run scripts appropriately.
- Due to the incompatibility issues (transformers library) comment in/out the imports of CANINE model/tokenizer in ```pfn.py``` script when the environment of CharacterBERT (environment 1) is used and vice versa. 
- For this study we also use code from the following repositories:
  - <a target="_blank" href="https://github.com/helboukkouri/character-bert">CharacterBERT</a> \[1\]
  - <a target="_blank" href="https://github.com/Coopercoppers/PFN">Partition Filter Network</a> \[2\]
  - <a target="_blank" href="https://github.com/christos42/CLDR_CLNER_models">CLDR & CLNER models</a> \[3\]


## References
```
[1] Hicham El Boukkouri, et al. 2020. CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations From Characters. In Proceedings of the 28th International Conference on Computational Linguistics. International Committee on Computational Linguistics, Barcelona, Spain (Online), 6903–6915. https://doi.org/10.18653/v1/2020.coling-main.609
[2] Zhiheng Yan, et al. 2021. A Partition Filter Network for Joint Entity and Relation Extraction. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 185–197. https://doi.org/10.18653/v1/2021.emnlp-main.17
[3] Christos Theodoropoulos, et al. 2021. Imposing Relation Structure in Language-Model Embeddings Using Contrastive Learning. In Proceedings of the 25th Conference on Computational Natural Language Learning. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 337–348. https://doi.org/10.18653/v1/2021.conll-1.27
```