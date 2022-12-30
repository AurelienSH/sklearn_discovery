from sklearn.model_selection import train_test_split
import clf,gensim.downloader, pretraitements, sys


w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
glove_vectors = gensim.downloader.load('glove-twitter-25')

if __name__ == "__main__":
    folder=sys.argv[1]

    files_classes=pretraitements.get_files_from_folder(folder)
    

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)