# Text-Summarization-using-TF-IDF
Text Summarization using TF-IDF (Term Frequency-Inverse Document Frequency)

The code follows the following logic in steps:
1. Provide an input text to the code. Since this is a big data project, Spark is used to input a file and map it into a python list.
2. Tokenize the sentences: All sentences are segregated on the basis of "full stop" punctuation mark using the function sent_tokenize().
3. The stop words are removed using NLTK.
4. Frequency matrix is generated which gives the count of all the words in a sentence using the function createFrequencyMatrix().
5. Next the Term Frequency is calculated which is defined as Term Frequency = (Number of times term t appears in a sentence)/(Total number of terms in the sentence). The function used is createTFMatrix().
6. As a prerequiste to calculating the Inverse Document Frequency, how many times word t appears in all the sentences is calculated using getSentencesPerWord() function. The resultant is returned as dictionary of word count. Total sentences in the document are simply calculated by applying len() function to sentences returned by sent_token() function in the second step.
7. The Inverse Document Frequency is calculated using log([Total number of documents]/[Number of documents in which term t appears]) formula using createIDFMatrix() function. The words in frequency matrix match the words dictionary returned by getSentencesPerWord(). Another dictionary within a dictionary is returned where the outer dictionary is the sentences(taken upto length 15 to save space) and the inner dictionary is the words with IDF value of each one.
8. The next step is just the multiplication of TF and IDF matrix to get the generate the TF-IDF matrix using createTFIDFMatrix() function.
9. The sentences are scored by adding the TF-IDF score of each word in a sentence divided by the number of sentences using the scoreSenteces() function.
