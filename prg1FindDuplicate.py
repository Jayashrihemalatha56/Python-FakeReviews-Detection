import pandas as pd
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

# Reading the data
#data = pd.read_csv("news.csv")
#data.head()
df = pd.read_table("reviewsNEW.csv", delimiter =",")#"reviews100.csv", delimiter =",")
print(df.head())
rowscount=len(df)
print(rowscount)

print('Enter news sentence to find REAL/FAKE:',end='')
newsline=input()
newsline=strip_non_ascii(newsline)
#newsline=(df.iat[2,1])
word=''
wordsnew=[]
for l in newsline.split():
    word= l.rstrip(',')
    word=word.rstrip('\'')
    word= word.lstrip(',')
    word=word.rstrip('\'')
    word= word.lstrip(':')
    word=word.rstrip(':')
    word= word.lstrip('\"')
    word=word.rstrip('\"')
    #print(word,end=' ')
    wordsnew.append(word)
line=''
word=''
cnt=0
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Function to calculate similarity percentage
def sentence_similarity(sentence1, sentence2):
    # Create a TfidfVectorizer to convert sentences to vectors
    vectorizer = TfidfVectorizer()

    # Convert the sentences to vectors
    vectors = vectorizer.fit_transform([sentence1, sentence2])

    # Calculate the cosine similarity between the two vectors
    similarity_matrix = cosine_similarity(vectors)

    # Extract the similarity score (second value in the first row)
    similarity_score = similarity_matrix[0][1]

    # Convert similarity to percentage with two decimal precision
    similarity_percentage = round(similarity_score * 100, 2)

    return similarity_percentage

# First sentence already defined
#sentence1 = "This is the first sentence."

for i in range(0,rowscount):
  #if i>2:
  #  exit()
  line=(df.iat[i,2])
  line=strip_non_ascii(line)
  words=[]
  for l in line.split():
    word= l.rstrip(',')
    word=word.rstrip('\'')
    word= word.lstrip(',')
    word=word.rstrip('\'')
    word= word.lstrip(':')
    word=word.rstrip(':')
    word= word.lstrip('\"')
    word=word.rstrip('\"')
    #print(word,end=' ')
    words.append(word)
  #print(words)
  #print(set(wordsnew))
  #print(set(words))
  #if len(set(wordsnew)) == len(set(words)):
     #print( len(set(wordsnew)))
     #print( len(set(words)))
  #res = len(set(wordsnew) & set(words)) / float(len(set(wordsnew) | set(words))) * 100
  #print(res)
  res= sentence_similarity(newsline,line)
  #print(res)
  if(res>75):
     print('Record ', i , '  Matched with : ', round(res,2) , ' %')
     print(df.iat[i,3])
     print("Similarity Percent:" + str(res))
  #else:
     #print("Similarity Percent:")
     #print(res)
     #print(df.iat[i,3])
#with open('glove.6B.50d.txt') as f:
#	for line in f:
#		values = line.split()
