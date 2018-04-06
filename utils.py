from sklearn.preprocessing import OneHotEncoder
import string
import matplotlib.pyplot as plt
import numpy as np

def getData(train_file = "./xtrain.txt", test_file = "./xtest.txt", labels_file = "./ytrain.txt"):
    train = [ row.strip() for row in open(train_file,"r").readlines() ]  
    test =  [ row.strip() for row in open(test_file,"r").readlines() ] 
    labels = [ row.strip() for row in open(labels_file,"r").readlines() ] 
    return train, test, labels

def getMaxLen(documents):
    max_len = 0
    for doc in documents:
        max_len = max(max_len, len(doc))
    return max_len

class OneHotEncode:
    def __init__(self):
        #fit over 26 (alphabets) + 1 (other) 
        self.enc = OneHotEncoder()
        self.enc.fit(np.array(range(0,27)).reshape(-1,1))
    
    def transform(self, x_train):
        x_train_enc = []
        for row in x_train:
            x_train_enc.append(self.enc.transform(row.reshape(-1,1)).toarray())
        return np.asarray(x_train_enc)

def get1hotY(labels, numClasses):
    labels = map(int, labels)
    y = np.zeros((len(labels), numClasses), dtype=np.int)
    for i in range(len(labels)):
        y[i][int(labels[i])] = 1
    return y
    
    
class Char_tokenizer:
    def __init__(self,documents):
            #create the dic of alphabet to number
            self.vocab = dict(zip(string.ascii_lowercase, range(1, 27)))
            self.max_len = getMaxLen(documents)
    
    #returns a matrix of dim: number of documents * max_len of document
    def transform(self, documents):
        num_docs = len(documents)
        X = np.zeros((num_docs, self.max_len), dtype=np.int)
        for i, document in enumerate(documents):
            for j, alpha in enumerate(document):
                X[i][j] = self.vocab[alpha]
        return X

    
#Reference: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def plotHistory(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    

def getNgramMaxLen(n):
    max_str_len = 453
    max_len = 0
    for i in range(n):
        max_len = max_len + max_str_len-n+1
    return max_len

def createDic(ngrams):
    counter = 0
    dic = {}
    for row in ngrams:
        for c in row:
            if c[0] not in dic:
                dic[c[0]] = counter
                counter = counter+1
    return dic

def encodeToNgrams(ngrams, dic, max_len):
    x = []
    for row in ngrams:
        _X = []
        for c in row:
           _X.append(dic[c[0]])
        x.append(_X)
    return x

def pad(X, max_len):
    _X = []
    for x in X:
        a = np.pad(x, (0, max_len - len(x)), 'constant', constant_values = (0))
        _X.append(a)
    return _X

def getNgrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        _ngrams = []
        for _n in range(1,n+1):
            for pos in range(1,len(sentence)-_n):
                _ngrams.append([sentence[pos:pos+_n]])
        ngrams.append(_ngrams)
    return ngrams

def getX(sentences, n, maxLen):
    ngrams = getNgrams(sentences, n)
    dic = createDic(ngrams)
    x = np.asarray(pad(encodeToNgrams(ngrams, dic, maxLen), maxLen))
    return (x, max(dic.values())+1)