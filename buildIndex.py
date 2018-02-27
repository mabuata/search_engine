from pyspark import SparkConf, SparkContext
import math

def main(sc):
    # reading input from HDFS. Reading the whole directory containing the documents. Assuming each document is small enough.
    textFile = sc.wholeTextFiles("/user/root/final/bbcsport/athletics") 
    N=textFile.count() # total number of documents
    # maping documents into pairs (word,doc) and prepare words
    rawWordList1 = textFile.flatMap(lambda (doc,contents): [(word.strip(".;,\'\"?!()").lower(),doc) for word in contents.split()])  
    rawWordList2 = rawWordList1.filter(lambda (word,doc):word!='')
    stopWords=sc.textFile("file:///root/lab/stopWords.txt").collect() # reading stop words from local file
    # wordList contains (word,doc) pairs that are not stop words
    wordList=rawWordList2.filter(lambda (word,doc):word not in stopWords ) # removing stop words
    #calculating term frequency (tf) of each word per doc
    wordCount = wordList.map(lambda (word,doc):((word,doc),1))# mapping/counting (1) to each word occurrence in a file
    TF = wordCount.reduceByKey(lambda v1, v2: v1+v2) #summming same word counts within a file i.e. calculating TF. TF is actually the inverted index ((word,file),tf)
    TF.persist()# TF is needed in more than one transformation
    DF=TF.map(lambda ((word,doc),v): (word,1)).reduceByKey(lambda v1, v2: v1+v2)# counting the document frequency of each word. Result RDD is ((word,file),df)
    # making word the key for the RDD elements TF
    TFWord=TF.map(lambda((word,doc),v):(word,(doc,v)))
    TF.unpersist()
    #combining tf and idf into one weight
    invertedIndex=TFWord.join(DF).map(lambda(word,((doc,tf),df)):(word,(doc,tf*math.log10(N/float(df)))))
    #preparing inverted index for output
    invertedIndexOutFormat= invertedIndex.map(lambda (word,(doc,tfidf)):word+" "+doc+" "+str(tfidf))
    #stroing inverted index in HDFS to be used by queries
    invertedIndexOutFormat.saveAsTextFile("/user/root/final/inverted.txt")

if __name__  == "__main__":
    conf = SparkConf().setAppName("buildIndex")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()



#spark-submit --master yarn-client --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m buildIndex.py

