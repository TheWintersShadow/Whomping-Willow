# Whomping-Willow

## How to use Whomping-Willow

Run:
`pip install WhompingWillow-1.0.0-py3-none-any.whl` 

In the Code:

```
from WhompingWillow import prep, lda

# To Prep the Data:
prepped_corpus = prep.prep_data(cleaned_corpus)

# Get Bag of Words and Dictionary
BoW, Dictionary = prep.build_BoW(prepped_corpus)

# Build LDA model
lda_model = lda.build_lda(Dictionary, BoW)

# Display LDA
lda.display_lda(lda_model, Dictionary, BoW)
```

### Prep_Data
```
Process the data
:param corpus: PD of the cleaned data. (Needs to have "data" column with the values for LDA.)
:return processed_docs: Processed PD
```

### Build_BOW
```
Build the Bog of Words and Gensim Dictionary.
:param processed_docs: PD of processed information.
:return BoW, id2word: Bag of Words and Gensim Dictionary
```

### Display_LDA
```
Display LDA with PyLDAVis
:param lda:
:param dictionary:
:param BoW:
:return nothing:
```

### Build_LDA
```
Build the LDA model.
:param dictionary:
:param BoW:
:return ldaModel: Gensim LDA Model
```
