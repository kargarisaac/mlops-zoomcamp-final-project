from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))

def preprocess(review):
    review_processed = review.lower()
    review_processed = review_processed.split()
    ps = PorterStemmer()
    review_processed =[ps.stem(i) for i in review_processed if not i in set(stopwords.words('english'))]
    review_processed =' '.join(review_processed)
    return review_processed

if __name__ == "__main__":
    print(preprocess("I am here."))