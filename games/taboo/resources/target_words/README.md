English words are taken from https://blog.prepscholar.com/100-funny-charades-ideas-for-a-hilarious-game

Frequency-based word lists (`games/taboo/utils/select_taboo_words.py`) are from https://www.kaggle.com/datasets/rtatman/english-word-frequency

### Instances for v1.5

First, we generate the instances for the experiments. 

If less than 3 related terms can be retrieved from Marriam Webster then we remove it from the selection and resample.

Then we go through the selected entries on the Marriam Webster Thesaurus website 
and select the 3 most highly ranked suggestions.  