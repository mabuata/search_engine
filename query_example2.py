from pyspark import SparkConf, SparkContext
import math

def main(sc):

    rawInvertedIndex = sc.textFile("/user/root/final/inverted.txt") # reading the inverted index from HDFS into RDD as string
    inverted1=rawInvertedIndex.map(lambda line:line.split())
    invertedIndex=inverted1.map(lambda (word,doc,tfidf): (word,doc,float(tfidf)))
    #invertedIndex=inverted1.map(lambda (word,doc,tfidf): (word[3:len(word)-1],doc[3:len(doc)-1],float(tfidf[0:len(tfidf)-2])))
    invertedIndex.persist()
    query=""
    while query!="-1":
        query = raw_input("Enter your query -1 to stop: ")
        n=int(raw_input("Enter # of maximum number of documents to retrieve: "))
        qWords=query.split()
        for i, qWord in enumerate(qWords):
            qWords[i]=qWord.lower().strip(".;,\'\"?!()")
            
        matchDocs=invertedIndex.filter(lambda (word,doc,tfidf):word in qWords)
        matchDocsScore1=matchDocs.map(lambda(word,doc,tfidf):(doc,tfidf)).reduceByKey(lambda tfidf1,tfidf2:tfidf1+tfidf2)
        matchDocsScore2=matchDocs.map(lambda(word,doc,tfidf):(doc,1)).reduceByKey(lambda v1,v2:v1+v2)
        matchDocScore=matchDocsScore1.join(matchDocsScore2).map(lambda (doc,(score1,score2)):(doc,score1*score2/float(len(qWords))))
        rerteivedDocs=matchDocScore.takeOrdered(n,key= lambda (doc,score):-score)
        for (doc,score) in rerteivedDocs:
            docText=sc.textFile(doc).collect()
            print "%s %s"% (doc,score)
            print('\n'.join(docText))
            print"-------------------------\n"

    invertedIndex.unpersist() 


if __name__  == "__main__":
    conf = SparkConf().setAppName("query")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()


#spark-submit --master yarn-client --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m query.py

