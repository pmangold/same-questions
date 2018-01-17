import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = {}
pairs_train = []
pairs_test = []
y_train = []

train_path = 'train.csv'
test_path = 'test.csv'

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

texts = {}
nb_lines = 1000
pairs_train, y_train = read_csv(train_path, texts, nb_lines = nb_lines)
pairs_test = read_csv(test_path, texts, labelled = False, nb_lines = nb_lines)
    
ids2ind, A = tfidf(texts)


N_train = len(pairs_train)
X_train = np.zeros((N_train, 3))
for i in range(len(pairs_train)):
    q1 = pairs_train[i][0]
    q2 = pairs_train[i][1]
    X_train[i,0] = cosine_similarity(A[ids2ind[q1],:], A[ids2ind[q2],:])
    X_train[i,1] = len(texts[q1].split()) + len(texts[q2].split())
    X_train[i,2] = abs(len(texts[q1].split()) - len(texts[q2].split()))


N_test = len(pairs_test)
X_test = np.zeros((N_test, 3))
for i in range(len(pairs_test)):
    q1 = pairs_test[i][0]
    q2 = pairs_test[i][1]
    X_test[i,0] = cosine_similarity(A[ids2ind[q1],:], A[ids2ind[q2],:])
    X_test[i,1] = len(texts[q1].split()) + len(texts[q2].split())
    X_test[i,2] = abs(len(texts[q1].split()) - len(texts[q2].split()))

clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

with open("submission_file.csv", 'w') as f:
    f.write("Id,Score\n")
    for i in range(y_pred.shape[0]):
        f.write(str(i)+','+str(y_pred[i][1])+'\n')

y_pred_train = clf.predict_proba(X_train)[:, 1]

def loss(y, p):
    N = y.shape[0]
    l = 0
    for i in range(N):
        if y[i] == 0:
            l -= np.log(1 - p[i])
        else:
            l -= np.log(p[i])
    return l / N

print "Score: ", loss(y_train, y_pred_train)
