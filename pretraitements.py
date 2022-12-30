import glob
import numpy as np
import gensim.downloader




def get_files_from_folder(folder):
    return [(file, file.split("/")[2]) for file in glob.glob(f"{folder}/*/*")]

def make_data(files_classes,dictClass):
    X=[]
    y=[]
    for file, class_ in files_classes:
        with open(file,"r") as f:
            X.append(f.readline())
        y.append(dictClass[class_])
    
    X=np.array(X)
    return X, y

def spacy_tokenize_noPunct_noSpaces(X,nlp):
    return [[token for token in nlp.tokenize(review) if not (token.is_punct or token.is_space)] for review in X]

def use_gensim_vectors(tokenized_X,word_vector):
    return[[word_vector[token] for token in review] for review in tokenized_X]