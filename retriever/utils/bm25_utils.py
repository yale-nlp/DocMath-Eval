from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce

from typing import List

def process_text(text: str) -> List[str]:
    ps = PorterStemmer()
    return [ps.stem(word) for word in word_tokenize(text.lower())]