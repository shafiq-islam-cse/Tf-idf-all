from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#text = ["The The The The The quick brown fox jumped over the lazy dog.",
#		"The dog.",
#		"The fox"]
text=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]

# create the transform
vectorizer = TfidfVectorizer()
#see tf idf images to know with expmle and equation 
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print("vectorizer.vocabulary_:\n",vectorizer.vocabulary_)
print("vectorizer.idf_:\n",vectorizer.idf_)


# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print("vector.shape:\n",vector.shape)

print("vector.toarray():\n",vector.toarray())