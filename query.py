from pyspark import SparkConf, SparkContext
import math
import sys
reload(sys)  
sys.setdefaultencoding('UTF8')

def main(sc,argv):
    query=argv[0:len(argv)-2]
    n=int(argv[len(argv)-1]) # number of documents to retrieve
    # reading the inverted index from HDFS into RDD as string
    rawInvertedIndex = sc.textFile("/user/root/final/inverted.txt")
    # reformatting the inverted index into the appropriate format of (word, doc, tfidf)
    inverted1=rawInvertedIndex.map(lambda line:line.split())
    invertedIndex=inverted1.map(lambda (word,doc,tfidf): (word,doc,float(tfidf)))
    invertedIndex.persist()
    #preprocess query terms
    for i, qWord in enumerate(query):
        query[i]=qWord.lower().strip(".;,\'\"?!()")
            
    #find inverted index elements that match one of the query terms/words
    matchDocs=invertedIndex.filter(lambda (word,doc,tfidf):word in query)
    #caclulating score1 (sum of tfidf per doc)
    matchDocsScore1=matchDocs.map(lambda(word,doc,tfidf):(doc,tfidf)).reduceByKey(lambda tfidf1,tfidf2:tfidf1+tfidf2)
    #caculating score2 (# of mutual terms between query and a doc)
    matchDocsScore2=matchDocs.map(lambda(word,doc,tfidf):(doc,1)).reduceByKey(lambda v1,v2:v1+v2)
    #combining score1 and score2 into final relevance score
    matchDocScore=matchDocsScore1.join(matchDocsScore2).map(lambda (doc,(score1,score2)):(doc,score1*score2/float(len(query))))
    #retrieving the top n relevant documents
    rerteivedDocs=matchDocScore.takeOrdered(n,key= lambda (doc,score):-score)
    #printing results to the user
    print(rerteivedDocs)
    for (doc,score) in rerteivedDocs:
        docText=sc.textFile(doc).collect()
        print "%s %s"% (doc,score)
        print('\n'.join(docText))
        print"-------------------------\n"

    invertedIndex.unpersist() 


if __name__  == "__main__":
    conf = SparkConf().setAppName("query")
    sc = SparkContext(conf = conf)
    main(sc,sys.argv[1:])
    sc.stop()


#spark-submit --master yarn-client --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m query.py query #of_retrived_documents
