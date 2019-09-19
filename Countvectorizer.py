import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

texts=["dog cat fish","dog cat cat","fish bird","bird"]
cv = CountVectorizer()
vector=cv.fit_transform(texts)
#https://stackoverflow.com/questions/27488446/how-do-i-get-word-frequency-in-a-corpus-using-scikit-learn-countvectorizer
print ("Word frequency in the document:",cv.vocabulary_)
print("vector.shape:",vector.shape)
print("type(vector):",type(vector))
print("Here Column is :different unique word in the corpus and \nRow is: Number of liene of tex in doc or corpus")
print("vector.toarray():\n",vector.toarray())

print(cv.get_feature_names())

#cv=CountVectorizer()
## this steps generates word counts for the words in your docs
#word_count_vector=cv.fit_transform(docs)
#word_count_vector.shape
#print("word_count_vector.shape:\n",word_count_vector.shape)
#tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
#tfidf_transformer.fit(word_count_vector)

print("vector.shape:\n",vector.shape)
print("vector:\n",vector)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(vector)
print("tfidf_transformer.fit(vector):\n",tfidf_transformer.fit(vector))

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["tf_idf_weights"])
df_idf.sort_values(by=['tf_idf_weights'])
print(" CountVectorizer()--using-(fit_transform(text)) and -tf_idf.sort_values(by=['tf_idf_weights']):\n",df_idf.sort_values(by=['tf_idf_weights']))



#output:{u'bird': 0, u'cat': 1, u'dog': 2, u'fish': 3}
'''
#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
#text = ["The quick brown fox jumped over the lazy dog."]
text = ["I am not enough qualified for this valuable post"]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit_transform(text)
# summarize
print("vectorizer.vocabulary_:\n",vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print("vector.shape:",vector.shape)
print("type(vector):",type(vector))
print("vector.toarray():",vector.toarray())
'''
