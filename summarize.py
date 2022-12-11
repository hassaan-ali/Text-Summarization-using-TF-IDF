import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col
from typing import List
import os

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


#IDF = log([Total number of documents]/[Number of documents in which term x appears])
def createIDFMatrix(frequencyMatrix, countDocPerWords, totalDocuments):
    IDFMatrix = {}

    for sentence, frequencyTable in frequencyMatrix.items():
        IDFTable = {}

        for word in frequencyTable.keys():
            IDFTable[word] = math.log10(totalDocuments/float(countDocPerWords[word]))

        IDFMatrix[sentence] = IDFTable
    
    return IDFMatrix

def createTFIDFMatrix(TFMatrix, IDFMatrix):
    TFIDFMatrix = {}

    for (sentence1, FreqTable1), (sentence2, FreqTable2) in zip(TFMatrix.items(), IDFMatrix.items()):
        TFIDFTable = {}

        for (word1, value1), (word2, value2) in zip(FreqTable1.items(), FreqTable2.items()): #Keys will be same for both the tables
            TFIDFTable[word1] = float(value1*value2)

        TFIDFMatrix[sentence1] = TFIDFTable
    
    return TFIDFMatrix


def scoreSentences(TFIDFMatrix) -> dict:
    sentenceScore = {}#defined as (sum of TF-IDF values of words in the sentence) / (Total number of words in the sentence)

    for sentence, TFIDFTable in TFIDFMatrix.items():
        sentenceScoreSum = 0 
        wordCount = len(TFIDFTable)#wordCount in the sentence
        if wordCount == 0:
            continue
        for word, TFIDFValue in TFIDFTable.items():
            sentenceScoreSum += TFIDFValue
        
        sentenceScore[sentence] = sentenceScoreSum/wordCount
    
    return sentenceScore

#Calculate the threshold by taking average of sentence score
def avgSentenceScore(sentenceValues) -> float:
    sumValues = 0

    for sentence in sentenceValues:
        sumValues += sentenceValues[sentence]
    
    return (sumValues/len(sentenceValues))

#Select the sentences whose score is more than the average.
def generateSummary(sentences, sentenceScores, supportThreshold):
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceScores and sentenceScores[sentence[:15]] >= supportThreshold:
            summary += " " + sentence
    
    return summary



if __name__ == "__main__":
    spark = SparkSession.builder.appName("TextSummarization").getOrCreate()
    path = os.getcwd()
    df = spark.read.text(os.getcwd() + '/input.txt', lineSep=".")
    cleanedSentences = df.withColumn("value", regexp_replace(col("value"), "[\n\r]", "" ))
    Sentences = (cleanedSentences.rdd.flatMap(lambda x:x).collect())
    FrequnceyMatrix = createFrequencyMatrix(sentences=Sentences)
    print("Frequency Matrix is: ", FrequnceyMatrix, '\n')
    TFMatrix = (createTFMatrix(FrequnceyMatrix))
    print("Term Frequency Matrix is: ",TFMatrix, '\n')
    countDocPerWords = (getSentencesPerWord(FrequnceyMatrix))
    totalDocuments = len(Sentences) #Number of sentences in the document.
    IDFMatrix = createIDFMatrix(FrequnceyMatrix, countDocPerWords=countDocPerWords, totalDocuments=totalDocuments)
    print("Inverse Document Frequencey is: ",IDFMatrix , '\n')
    TF_IDF_Matrix = createTFIDFMatrix(TFMatrix, IDFMatrix)
    print("TF-IDF Matrix is: ", TF_IDF_Matrix, '\n')
    sentenceScores = (scoreSentences(TF_IDF_Matrix))
    print("Sentence values are: ", sentenceScores, '\n')
    supportThreshold = avgSentenceScore(sentenceScores)
    print("The support threshold is: ", supportThreshold, '\n')
    summary = (generateSummary(Sentences, sentenceScores, supportThreshold))
    with open(path + "/output.txt", mode ='w') as file:
        file.write(summary)
    



