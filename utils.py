import nltk

def tokenize(sequence):
    '''
    Inputs:
        caption: string
    Returns:
        Tokenized sentence (list of strings)
        All the tokens are converted to lower-case
    '''
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens
