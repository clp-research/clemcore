
# Guess What?

Implemented by: Josefa de la Luz Costa Rojo & Melisa Özdemir

We used two different datasets to create the instances of the games, one that mixed abstract and concrete words and another one from which we only used abstract words. The full datasets and all the information related to them can be found below: 

>Castro, N., Curley, T., & Hertzog, C. (2021). Category norms with a cross-sectional sample of adults in the United States: Consideration of cohort, age, and historical effects on semantic categories. Behavior Research Methods, 53(2), 898–917. https://doi.org/10.3758/s13428-020-01454-9

>Banks, B., & Connell, L. (2022). Category Production Norms for 117 Concrete and Abstract Categories. OSF. https://osf.io/jgcu6

This game aims to assess whether clems can develop efficient information-seeking strategies to narrow down possible options quickly, while considering how semantic relationships between words impact this process. With this purpose, we created candidate lists of 8 words with varying degrees of semantic similarity, grouped into distinct categories and subcategories.
The goal is to see if the model can identify the shared property among these words and ask strategic questions based on that understanding. This ability is evaluated using a Quality Score, where fewer turns to guess the correct word indicate a more effective search strategy.

## Experiments
According to the semantic relationship of the words there are 3 experiments in the game that goes from less related to more, and they are structured as follows: 

- **Level 1**: 4 categories, each containing 2 subcategories, with 1 word per subcategory.
- **Level 2**: 2 categories, each containing 2 subcategories, with 2 words per subcategory.
- **Level 3**: 1 category, containing 4 subcategories, with 2 words per subcategory.
- **Abs Level 1**: 4 categories, each containing 2 subcategories, with 1 word per subcategory.(abstract words)
- **Abs Level 2**: 2 categories, each containing 2 subcategories, with 2 words per subcategory. (abstract words)
- **Abs Level 3**: 1 category, containing 4 subcategories, with 2 words per subcategory. (abstract words)


## Scores
Apart from the common metrics of the framework the game also measures the following:

- **Speed**: How quickly the correct guess was made relative to the ideal number of turns for each level (lower bound).
- **Quality Score**: Speed
- **Invalid Content response**: Counts the number of content invalid responses from each player.
- **Invalid Format response**: Counts the number of invalid form responses from each player.