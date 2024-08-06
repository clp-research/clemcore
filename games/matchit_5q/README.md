# MatchIt - 5q version

MatchIt is a multimodal dialogue game where two players have to come to an agreement whether the image that each of them gets as input is the same or not.
This is the 5q version of matchit, meaning that each player can ask the other exactly five questions about the other image.

# How to sample different number of instances

The instancegenerator.py generates instances by sampling N examples (N $\leq$ 161) per difficulty (same, similar and different image) from the lists of image pairs in resources/imagepairs. N and other version parameters (such as the number of questions) are declared in the instancegenerator file. If N should be larger or other image pairs are wanted, see [here](resources/additional/README.md).


# Image pair sampling rationale

The image pairs for the "different"- category are selected by taking pairs with the lowest Jaccard-Indices. 
The image pairs for the "similar" category are chosen by a combination of Jaccard-Index and CLIP-Score. First, only pairs with a Jaccard-Index above 0.22 are chosen of which then the CLIP-Score is calculated. From experience it seems that then choosing pairs with CLIP-Scores above 0.9 yields very good results, but above 0.8 should also be fine as long as the pairs are inspected before use. 

# Resources
* image_pairs: contains pairs of images used by imagegenerator.py
    * [similar_images.csv](resources/image_pairs/similar_images.csv): hand curated list of 161 similar image pairs derived from 2000_largest_clipscore.csv (criteria: Jaccard score higher than 0.22, CLIP score higher than 0.8 and images should be similar, but easily discriminable)
    * [different_images.csv](resources/image_pairs/different_images.csv): random sample of 1000 image pairs out of all pairs with a Jaccard-Index below 0.05 from a list of image pairs out of 60000 images which was is ~6 million entries long. For faster instance generation this subsample exists.
* additional
    * necessary information and files for generating more image pairs
    * [1000_largest_of_60000_jac.csv](resources/additional/1000_largest_of_60000_jac.csv): a list of the 1000 image pairs with the highest Jaccard-Indices out of ~1.8 billion image pairs (all possible pairs out of a 60000 subset of the Visual Genome data set)
    * [largest_clipscore.csv](resources/additional/largest_clipscore.csv): 1572 image pairs with CLIP-scores between  0.31 and  0.95 and Jaccard indices above 0.22
* initial_prompts: contains all necessary prompt templates
