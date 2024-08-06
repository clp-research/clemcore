# How to sample new picture pairs alltogether

If you want to generate more than 161 instances (current number of similar image pairs), you can use imagepairs.py by running

```python3 imagepairs.py k n```

where:
-  ```k``` is the number of pictures from Visual Genome that are used in the process and 
- ```n``` the number of pairs that will be saved in the output. They are produced by taking the top n pairs with the highest Jaccard similarities of Visual Genome object labels and their respective attributes for each image was calculated.



**Attention**:  this script downloads all images from the Visual Genome Dataset for the n pairs in order to get the CLIP-score! Depending on the number of samples, this can take some time - be sure that you want this and have enough memory space. 

## Output
- n_largest.csv : dataframe of n image pairs, with respective Jaccard similarities
- n_largest_clipscore.csv : same dataframe as above with added CLIP score values
- picturesk_n: folder with the downloaded image files from the n pairs
- k_pics_sample_jaccards: .zip-file with Jaccard similarities for all possible pairs from k images

From n_largest_clipscore.csv similar pairs can be sampled above chosen thresholds. I suggest inspecting the resulting pairs before use.

Different pairs can be sampled from the lower end from k_pics_sample_jaccards. Due to the nature of the set-similarity package, no Jaccard-similarities, that are 0 exactly can be returned (see this [issue](https://github.com/ekzhu/SetSimilaritySearch/issues/19)). Nonetheless, image pairs with Jaccard-Indices below 0.05 should be safe to use since less than 5% of the contents (attributes and/or objects) are shared. If the file is too large, 

## Requirements

The following files need to be downloaded from from Visual Genome (https://homes.cs.washington.edu/~ranjay/visualgenome/api.html, Version 1.2 of dataset completed as of August 29, 2016) and saved in the same folder as imagepairs.py:
- [image_data.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip) (17.62 MB)
- [attributes.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip) (462.56 MB)



Package requirements are specified in [requirements.txt](requirements.txt), [SetSimilaritySearch package](https://github.com/ekzhu/SetSimilaritySearch) has also to be installed
