# Based on this here:
# http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun"
)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print tfidf_matrix.shape

res = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print res

cos_sim = res[0][2]
angle_in_radians = math.acos(cos_sim)
print math.degrees(angle_in_radians)


