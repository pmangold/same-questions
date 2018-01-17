# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import numpy as np
import unidecode

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk.corpus
import nltk.tokenize.punkt
import string
from nltk.stem.snowball import SnowballStemmer

#nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

save = False

texts = {}
pairs_train = []
pairs_test = []
y_train = []

train_path = 'train.csv'
test_path = 'test.csv'
sub_path = 'submissions.csv'

def read_csv(path, texts, nb_lines = None, labelled = True):
    pairs = []
    y = []
    read_lines = 0
    
    with open(path,'r') as f:
        for line in f:
            if nb_lines != None and read_lines >= nb_lines:
                break
            read_lines += 1
            
            l = line.split(',')
            if l[1] not in texts:
                texts[l[1]] = l[3]
            if l[2] not in texts:
                if labelled:
                    texts[l[2]] = l[4]
                else:
                    texts[l[2]] = l[4][:-1]                    

            pairs.append([l[1],l[2]])

            if labelled:
                y.append(int(l[5][:-1])) # [:-1] is just to remove formatting at the end

    if labelled:
        return pairs, np.array(y)
    else:
        return pairs

def tfidf(texts):
    ids2ind = {} # will contain the row idx of each unique text in the TFIDF matrix 
    for qid in texts:
        ids2ind[qid] = len(ids2ind)

    vec = TfidfVectorizer()
    A = vec.fit_transform(texts.values())

    return ids2ind, A

def compute_features(pairs, A, ids2ind):
    N = len(pairs)
    X = np.zeros((N, 3))
    for i in range(len(pairs)):
        q1 = pairs[i][0]
        q2 = pairs[i][1]
        X[i,0] = cosine_similarity(A[ids2ind[q1],:], A[ids2ind[q2],:])
        X[i,1] = len(texts[q1].split()) + len(texts[q2].split())
        X[i,2] = abs(len(texts[q1].split()) - len(texts[q2].split()))

    return N, X

def save_submission(sub_path, y):
    with open(sub_path, 'w') as f:
        f.write("Id,Score\n")
        for i in range(y_pred.shape[0]):
            f.write(str(i)+','+str(y_pred[i][1])+'\n')

        
        
def loss(y, p):
    N = y.shape[0]
    l = 0
    for i in range(N):
        if y[i] == 0:
            l -= np.log(1 - p[i])
        else:
            l -= np.log(p[i])
    return l / N


def preprocess_line(line, stemmer = stemmer):
    line = line.lower().translate(None, string.punctuation)
    l = " ".join(map(stemmer.stem, line.split(" ")))
    
    return l

def preprocess_texts(texts):
    for i in texts.keys():
        texts[i] = preprocess_line(texts[i])

texts = {}
nb_lines = None
pairs_train, y_train = read_csv(train_path, texts, nb_lines = nb_lines)
pairs_test = read_csv(test_path, texts, labelled = False, nb_lines = nb_lines)

preprocess_texts(texts)
 
ids2ind, A = tfidf(texts)

N_train, X_train = compute_features(pairs_train, A, ids2ind)
N_test, X_test = compute_features(pairs_test, A, ids2ind)


clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

if save:
    save_submission(sub_path, y_pred)

y_pred_train = clf.predict_proba(X_train)[:, 1]

print("Score: ", loss(y_train, y_pred_train))
