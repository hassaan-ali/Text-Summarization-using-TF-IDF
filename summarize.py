import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math

#Uncomment the below if running the code for the first time to download specific libraries from NLTK.
#nltk.download('punkt')
#nltk.download('stopwords')

#createFrequencyMatrix will return a dictionary with sentences as keys and count of number of times the word appears in that sentence.

def createFrequencyMatrix(sentences):
    frequencyMatrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sentence in sentences:
        frequencyTable = {}
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            
            if word in frequencyTable:
                frequencyTable[word] += 1
            else:
                frequencyTable[word] = 1
            
        frequencyMatrix[sentence[:15]] = frequencyTable

    return frequencyMatrix

# Term Frequency = (Number of times term t appears in a sentence)/(Total number of terms in the sentence)
# Here a sentence is considered as a document
def createTFMatrix(frequencyMatrix):
    TFMatrix = {}

    for sentence, words in frequencyMatrix.items():
        TFTable = {}

        wordCountSentence = len(words)

        for word, wordCount in words.items():
            TFTable[word] = wordCount/wordCountSentence
        
        TFMatrix[sentence] = TFTable
    
    return TFMatrix

#how many times a word appears in all the sentences
def getSentencesPerWord(frequencyMatrix):
    coutWordPerDoc = {}

    for sentence, words in frequencyMatrix.items():
        for word, wordCount in words.items():
            if word in coutWordPerDoc:
                coutWordPerDoc[word] += 1
            else:
                coutWordPerDoc[word] = 1
    
    return coutWordPerDoc


#IDF = ln([Total number of documents]/[Number of documents in which term x appears])
def createIDFMatrix(frequencyMatrix, countDocPerWords, totalDocuments):
    IDFMatrix = {}

    for sentence, frequencyTable in frequencyMatrix.items():
        IDFTable = {}

        for word in frequencyTable.keys():
            IDFTable[word] = math.log10(totalDocuments/float(countDocPerWords[word]))

        IDFMatrix[sentence] = IDFTable
    
    return IDFMatrix



if __name__ == "__main__":
    text = "When I do count the clock that tells the time. And see the brave day sunk in hideous night, When I behold the violet past prime,"
    sentences = sent_tokenize(text) #All the sentences segregated based on full stop.
    FrequnceyMatrix = createFrequencyMatrix(sentences=sentences)
    print("Frequency Matrix is: ", FrequnceyMatrix, '\n')
    TFMatrix = (createTFMatrix(FrequnceyMatrix))
    print("Term Frequency Matrix is: ",TFMatrix, '\n')
    countDocPerWords = (getSentencesPerWord(FrequnceyMatrix))
    totalDocuments = len(sentences) #Number of sentences in the document.
    print(createIDFMatrix(FrequnceyMatrix, countDocPerWords=countDocPerWords, totalDocuments=totalDocuments))
