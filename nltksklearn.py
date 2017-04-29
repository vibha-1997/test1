import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)
word_features=list(all_words.keys())[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)

    return features
##print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set=featuresets[:1900]
test_set=featuresets[1900:]
##classifier=nltk.NaiveBayesClassifier.train(training_set)
classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()


print(" original classifier %,",(nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(15)


MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("multinomial %=",(nltk.classify.accuracy(MNB_classifier,test_set))*100)

###Gaussian
##G_classifier=SklearnClassifier(GaussianNB())
##G_classifier.train(training_set)
##print("Gaussian %=",(nltk.classify.accuracy(G_classifier,test_set))*100)

B_classifier=SklearnClassifier(BernoulliNB())
B_classifier.train(training_set)
print("Bernoulli %=",(nltk.classify.accuracy(B_classifier,test_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  B_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)

print("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence %:",voted_classifier.confidence(test_set[0][0])*100)
print("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence %:",voted_classifier.confidence(test_set[1][0])*100)
print("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence %:",voted_classifier.confidence(test_set[2][0])*100)
print("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence %:",voted_classifier.confidence(test_set[3][0])*100)
print("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence %:",voted_classifier.confidence(test_set[4][0])*100)
print("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence %:",voted_classifier.confidence(test_set[5][0])*100)
save_classifier=open("naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()
