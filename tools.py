from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

import numpy as np


def loss(y, p):
    N = y.shape[0]
    l = 0
    ignored = 0
    for i in range(N):
        if (y[i] == 0 and p[i][1] == 1.) or (y[i] == 1 and p[i][1] == 0.):
            ignored += 1
        else:
            if y[i] == 0:
                l -= np.log(1 - p[i][1])
            else:
                l -= np.log(p[i][1])
    return l / (N - ignored)


def compute_score(clf, X, y):
    scorer = make_scorer(loss, greater_is_better = True, needs_proba = True)
    return scorer(clf, X, y)


def compute_cv_score(clf, X, y, cv = 5):
    loss_scorer = make_scorer(loss, greater_is_better = True, needs_proba = True)
    scoring = {'acc': 'accuracy',
               'loss': loss_scorer}
    scores = cross_validate(clf, X, y, scoring=scoring,
                             cv=cv, return_train_score=True)  
    
    return scores

def print_line(text, acc, loss, number = True):
    if number:
        acc = '%0.2f' % acc
        loss = '%0.2f' % loss
    print(color.BOLD +
              color.YELLOW + text + "\t\t" + 
              color.BLUE + acc + "\t" + 
              color.GREEN + loss + color.END)
    
def print_score(clf, X, y, cv = 5):
    cv_scores = compute_cv_score(clf, X, y, cv = cv)
    acc_color = "blue"
    loss_color = "green"
    layout_color = "yellow"
    
    for i in range(cv):
        print(color.BOLD + color.YELLOW + "CV Fold %i" % i + color.END)
        print_line("", "acc", "loss", False)
        print_line("train", cv_scores['train_acc'][i], cv_scores['train_loss'][i])
        print_line("test", cv_scores['test_acc'][i], cv_scores['test_loss'][i])
        print()
        
        
    print(color.BOLD + color.YELLOW + "Bagged scores" + color.END)
    print_line("train", np.mean(cv_scores['train_acc']), np.mean(cv_scores['train_loss'][i]))
    print_line("test", np.mean(cv_scores['test_acc']), np.mean(cv_scores['test_loss']))
    
def save_submission(sub_path, y):
    with open(sub_path, 'w') as f:
        f.write("Id,Score\n")
        for i in range(y.shape[0]):
            f.write(str(i)+','+str(y[i][1])+'\n')

def read_csv(path, texts, nb_lines = None, labelled = True):
    pairs = []
    y = []
    read_lines = 0
    
    with open(path,'r', encoding = 'utf8') as f:
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

# for displaying score
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
