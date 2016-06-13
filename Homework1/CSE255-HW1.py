# I am learning GIT

# coding: utf-8
# In[1]:

from pyspark import SparkContext
sc = SparkContext()

# In[2]:

textRDD = sc.newAPIHadoopFile('/data/Moby-Dick.txt',
                              'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                              'org.apache.hadoop.io.LongWritable',
                              'org.apache.hadoop.io.Text',
                               conf={'textinputformat.record.delimiter': "\r\n\r\n"}) \
            .map(lambda x: x[1])

sentences=textRDD.flatMap(lambda x: x.split(". "))


# In[5]:

## Foundamental String processing:
##(1) to lower case 
##(2) replace the new line 
##(3) remove punctuations. 
import string

def nopunc(x):
    no_punct = ""
    for char in x:
        if char not in string.punctuation:
            no_punct = no_punct+char
    return no_punct

senfinal = sentences.map(lambda x: x.lower()) .map(lambda x: x.replace("\r\n", " ")) .map(lambda x: nopunc(x))


# In[6]:

def printOutput(n,freq_ngramRDD):
    top=freq_ngramRDD.take(5)
    print '\n============ %d most frequent %d-grams'%(5,n)
    print '\nindex\tcount\tngram'
    for i in range(5):
        print '%d.\t%d: \t"%s"'%(i+1,top[i][0],' '.join(top[i][1]))


# In[7]:

## Countings: 
## (1) seperate the words in each sentence. 
## (2) In the for-loop, make the n-gram and counting pairs 
## (3) using the reduceByKey to add all counts. 
## (4) sort by the counts 
## (5) print them out
sen = senfinal.map(lambda x: x.split())
for n in range(1,6):
    freq_ngramRDD = sen.flatMap(lambda x: [(tuple(x[i:(i+n)]),1) for i in range(len(x)+1-n)])     .reduceByKey(lambda x,y:x+y)     .sortBy(lambda x:x[1], False)     .map(lambda x: (x[1], x[0]))
    printOutput(n,freq_ngramRDD)


# In[ ]:



