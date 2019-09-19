#https://github.com/kavgan/nlp-in-practice/blob/master/tfidftransformer/TFIDFTransformer%20vs.%20TFIDFVectorizer.ipynb
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# this is a very toy example, do not try this at home unless you want to understand the usage differences
docs=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]
cv=CountVectorizer()

# this steps generates word counts for the words in your docs

word_count_vector=cv.fit_transform(docs)

word_count_vector.shape
print("word_count_vector.shape:\n",word_count_vector.shape)
print("word_count_vector:\n",word_count_vector)
print("Here Column is :different unique word in the corpus and \nRow is: Number of liene of tex in doc or corpus")
print("word_count_vector.toarray():\n",word_count_vector.toarray())


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

print("tfidf_transformer.fit(word_count_vector):\n",tfidf_transformer.fit(word_count_vector))
# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["tf_idf_weights"])
df_idf.sort_values(by=['tf_idf_weights'])

print(" CountVectorizer()--using fit-transform-(tfidf_transformer.idf_(text)) and -tf_idf.sort_values(by=['tf_idf_weights']):\n",df_idf.sort_values(by=['tf_idf_weights']))

count_vector=cv.transform(docs)

tf_idf_vector=tfidf_transformer.transform(count_vector)
print("tf_idf_vector:\n",tf_idf_vector)
feature_names = cv.get_feature_names()

#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]

#print the scores
df = pd.DataFrame(first_document_vector.T.toarray(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
print("\n********This works only for first row of the tf_idf_vector and others row value is assigned as 0 *********\n")
print("CountVectorizer()-using --(transform(text)) df.sort_values(by=[tfidf],ascending=False):\n",df.sort_values(by=["tfidf"],ascending=False))

print("\n********Tfidfvectorizer*********\n")
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
#print("tfidf_vectorizer_vectors.idf_:\n",tfidf_vectorizer_vectors.idf_)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
df1 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df1.sort_values(by=["tfidf"],ascending=False)
print("CountVectorizer()-using --(fit-transform(text)) df1.sort_values(by=[tfidf],ascending=False):\n",df1.sort_values(by=["tfidf"],ascending=False))
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

# just send in all your docs here
fitted_vectorizer=tfidf_vectorizer.fit(docs)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
df2 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df2.sort_values(by=["tfidf"],ascending=False)
print("CountVectorizer()-using --(fit.(text)) df2.sort_values(by=[tfidf],ascending=False):\n",df2.sort_values(by=["tfidf"],ascending=False))
