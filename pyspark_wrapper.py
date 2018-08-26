#!/usr/bin/python
# encoding: utf-8

from pyspark.sql import SparkSession

class pyspark_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def get_sparksession(self, master,appname):
        '''
        获取pyspark.sql.session.SparkSession的实例
        @param master: a url for spark master
            如：local,
        @appname: saprk application的名字
            一个spark应用的层次：application含多个job，job含多个stage，stage含多个task
        '''
        spark = SparkSession.builder \
             .master("local") \
             .appName("Word Count") \
             .getOrCreate()
        return spark
    @classmethod
    def get_sparkcontext(self, sparksession):
        '''
        通过sparksession获取pyspark.context.SparkContext的实例
        @sparksession: sparksession实例
        '''
        sc = sparksession.sparkContext
        return sc
    '''
    ###########################################
    DataFrame 操作
    生成DF的几种方式
    * 通过读取文件生成 spark.read.csv
    * spark.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
    ###########################################
    '''
    def csv(self, spark, path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None):
        '''
        Loads a CSV file and returns the result as a  :class:`DataFrame`.
        @spark: pyspark.sql.session.SparkSession的实例
        df = spark.read.csv("file:///home/ian/code/data/sparkml/wordcount", sep='|', header=False)
        '''
        return spark.read.csv(path, schema, sep, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None)
    def createDataFrame(self, spark, data, schema=None, samplingRatio=None, verifySchema=True):
        '''
        Creates a :class:`DataFrame` from an :class:`RDD`, a list or a :class:`pandas.DataFrame`.
        '''
        return spark.createDataFrame(data, schema, samplingRatio=None, verifySchema=True)
    
    def show(n=20, truncate=True, vertical=False):
        '''
        Prints the first ``n`` rows to the console.
        '''        
        return df.show(n=20, truncate=True, vertical=False)
    def first(self,df):
        '''
        Returns the first row as a :class:`Row`.

        >>> df.first()
        Row(age=2, name='Alice')
        '''        
        return df.first()
    def withColumnRenamed(self,df, existing, new):
        '''
        修改df的列名，注意这会复制出一个新的df，而不是在原df上做修改
        Returns a new :class:`DataFrame` by renaming an existing column.
        This is a no-op if schema doesn't contain the given column name.
        '''        
        return df.withColumnRenamed(existing, new)
    def sort(self,df,*cols, **kwargs):
        '''
        Returns a new :class:`DataFrame` sorted by the specified column(s).
        df3.sort(df3['_2'].desc())
        '''        
        return df.sort(*cols, **kwargs)
    
    '''
    ###########################################
    RDD 操作
    生成RDD有三种方式
    * 通过读取文件生成 textFile
    * parallelize
    * 通过父RDD经过transformation算子转换得到
    ###########################################
    '''
    def textFile(self, sc, name, minPartitions=None, use_unicode=True):
        '''
        ### 从文件系统的文件生成rdd
        * 默认是存hdfs中的文件路径获取  
        如/home/ian/bigdl.log相当于hdfs://h1:9000/home/ian/bigdl.log  
        * 如果要从本地文件系统获取文件需要加上file://，如file:///home/ian/test.py
        @sc: pyspark.context.SparkContext的实例
        @name: 读取的文件url
        Read a text file from HDFS, a local file system (available on all
        nodes), or any Hadoop-supported file system URI, and return it as an
        RDD of Strings.
        
        If use_unicode is False, the strings will be kept as `str` (encoding
        as `utf-8`), which is faster and smaller than unicode. (Added in
        Spark 1.2)
        '''
        return sc.textFile(name, minPartitions, use_unicode)
    
    def parallelize(self, sc, c, numSlices=None):
        '''
        把本地的python collection转换为分布式的rdd对象
        @sc: pyspark.context.SparkContext的实例
        @c: 本地的python collection
        @numSlices： rdd的partition数，默认为分配到spark的cpu core的数量（待验证）
        Distribute a local Python collection to form an RDD. Using xrange
        is recommended if the input represents a range for performance.
        '''
        return sc.parallelize(c, numSlices)
    def getNumPartitions(self,rdd):
        '''
        Returns the number of partitions in RDD
        '''
        return rdd.getNumPartitions()
    
    '''
    ###########################################
    RDD transformation算子
    ###########################################
    '''
    def map1(self,rdd,f, preservesPartitioning=False):
        '''
        Return a new RDD by applying a function to each element of this RDD.

        >>> rdd = sc.parallelize(["b", "a", "c"])
        >>> sorted(rdd.map(lambda x: (x, 1)).collect())
        [('a', 1), ('b', 1), ('c', 1)]
        '''
        return rdd.map(f,preservesPartitioning)
    def flatMap(self,rdd,f, preservesPartitioning=False):
        '''
        分两步，先执行map，通过f函数把元素x映射为一个集合，如(1,x)
        在执行flatten操作，把集合扁平化，即(1,x)=>1,x
        Return a new RDD by first applying a function to all elements of this
        RDD, and then flattening the results.

        >>> rdd = sc.parallelize([2, 3, 4])
        >>> sorted(rdd.flatMap(lambda x: range(1, x)).collect())
        [1, 1, 1, 2, 2, 3]
        >>> sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect())
        [(2, 2), (2, 2), (3, 3), (3, 3), (4, 4), (4, 4)]
        '''
        return rdd.flatMap(f,preservesPartitioning)
    def glom(self,rdd):
        '''
        Return an RDD created by coalescing（合并） all elements within each partition
        into a list.
        @return: pyspark.rdd.PipelinedRDD的实例
        >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
        >>> rdd.collect()
        [1, 2, 3, 4]
        >>> rdd.glom().collect()
        [[1, 2], [3, 4]]
        '''
        return rdd.glom()
    def partitionBy(self, rdd, numPartitions, partitionFunc=<function portable_hash at 0x7fd5db73c598>):
        '''
        把rdd按照指定的partitioner重新划分生成新的partitions, 元素的值保持不变
        Return a copy of the RDD partitioned using the specified partitioner.

        >>> pairs = sc.parallelize([1, 2, 3, 4, 2, 4, 1]).map(lambda x: (x, x))
        >>> sets = pairs.partitionBy(2).glom().collect()
        >>> sets
        [[(2, 2), (4, 4), (2, 2), (4, 4)], [(1, 1), (3, 3), (1, 1)]]
        '''
        return rdd.partitionBy(numPartitions, partitionFunc)
    def groupBy(self, rdd, f, numPartitions=None, partitionFunc=<function portable_hash at 0x7fd5db73c598>):
        '''
        Return an RDD of grouped items.
        通过示例可知，rdd的元素不是(key,value)的形式，结合示例好好理解一下
        >>> rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
        >>> result = rdd.groupBy(lambda x: x % 2).collect()
        >>> sorted([(x, sorted(y)) for (x, y) in result])
        [(0, [2, 8]), (1, [1, 1, 3, 5])]
        '''
        return rdd.groupBy(f, numPartitions, partitionFunc)
    def groupByKey(self, rdd, numPartitions, partitionFunc=<function portable_hash at 0x7fd5db73c598>):
        '''
        把rdd按照指定的partitioner重新划分生成新的partitions, 元素按照key进行合并，把相同的key的值组成一个iterator
        Group the values for each key in the RDD into a single sequence.
        Hash-partitions the resulting RDD with numPartitions partitions.

        .. note:: If you are grouping in order to perform an aggregation (such as a
            sum or average) over each key, using reduceByKey or aggregateByKey will
            provide much better performance.

        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.groupByKey().mapValues(len).collect())
        [('a', 2), ('b', 1)]
        >>> sorted(rdd.groupByKey().mapValues(list).collect())
        [('a', [1, 1]), ('b', [1])]
        '''
        return rdd.groupByKey(numPartitions, partitionFunc)
    def reduceByKey(self, rdd, f, numPartitions=None, partitionFunc=<function portable_hash at 0x7fd5db73c598>):
        '''
        Merge the values for each key using an associative and commutative reduce function.

        This will also perform the merging locally on each mapper before
        sending results to a reducer, similarly to a "combiner" in MapReduce.

        Output will be partitioned with C{numPartitions} partitions, or
        the default parallelism level if C{numPartitions} is not specified.
        Default partitioner is hash-partition.

        >>> from operator import add
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.reduceByKey(add).collect())
        [('a', 2), ('b', 1)]
        '''
        return rdd.reduceByKey(f, numPartitions, partitionFunc)
    @classmethod
    def union(self, rdd1, rdd2):
        '''
        将两个rdd合并，注意只是简单的合并，partition不会改变，合并后的rdd的partition数是两个rdd的partition数之和
        Return the union of this RDD and another one.
        >>> rdd.glom().collect()
        [[1, 2], [3, 4]]
        >>>rdd1.glom().collect()
        [[(2, 2), (4, 4), (2, 2), (4, 4)], [(1, 1), (3, 3), (1, 1)]]
        >>rdd1.union(rdd).glom().collect()
        [[(2, 2), (4, 4), (2, 2), (4, 4)], [(1, 1), (3, 3), (1, 1)], [1, 2], [3, 4]]
        '''
        return rdd1.union(rdd2)
    @classmethod
    def join(self, rdd1, rdd2, numPartitions=None):
        '''
        把rdd1和rdd2中具有相同的key的元素进行join操作，见示例
        Return an RDD containing all pairs of elements with matching keys in
        C{self} and C{other}.

        Each pair of elements will be returned as a (k, (v1, v2)) tuple, where
        (k, v1) is in C{self} and (k, v2) is in C{other}.

        Performs a hash join across the cluster.

        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2), ("a", 3)])
        >>> sorted(x.join(y).collect())
        [('a', (1, 2)), ('a', (1, 3))]
        '''
        return rdd1.join(rdd2, numPartitions)
    '''
    ###########################################
    RDD action算子
    ###########################################
    '''
    @classmethod
    def collect(self, rdd):
        '''
        Return a list that contains all of the elements in this RDD.

        .. note:: This method should only be used if the resulting array is expected
            to be small, as all the data is loaded into the driver's memory.
        @rdd: pyspark.rdd.RDD对象
        '''
        return rdd.collect()
    @classmethod
    def count(self, rdd):
        '''
        Return the number of elements in this RDD.
        @rdd: pyspark.rdd.RDD对象
        '''
        return rdd.count()
    @classmethod
    def reduce(self, rdd, f):
        '''
        Reduces the elements of this RDD using the specified commutative and
        associative binary operator. Currently reduces partitions locally.

        >>> from operator import add
        >>> sc.parallelize([1, 2, 3, 4, 5]).reduce(lambda a,b: a+b)
        15
        >>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
        10
        >>> sc.parallelize([]).reduce(add)
        Traceback (most recent call last):
            ...
        ValueError: Can not reduce() empty RDD
        '''
        return rdd.reduce(f)
    @classmethod
    def countByKey(self, rdd):
        '''
        Count the number of elements for each key, and return the result to the
        master as a dictionary.

        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.countByKey().items())
        [('a', 2), ('b', 1)]
        '''
        return rdd.countByKey()
    @classmethod
    def foreach(self,rdd, f):
        '''
        Applies a function to all elements of this RDD.
        >>> def f(x): print(x)
        >>> sc.parallelize([1, 2, 3, 4, 5]).foreach(f)
        '''
        return rdd.foreach(f)
    @classmethod
    def saveAsHadoopFile(self, path, outputFormatClass, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, conf=None, compressionCodecClass=None):
        '''
        Output a Python RDD of key-value pairs (of form C{RDD[(K, V)]}) to any Hadoop file
        system, using the old Hadoop OutputFormat API (mapred package). Key and value types
        will be inferred if not specified. Keys and values are converted for output using either
        user specified converters or L{org.apache.spark.api.python.JavaToWritableConverter}.
        >>> rdd1.saveAsHadoopFile('/rdd1',"org.apache.hadoop.mapred.SequenceFileOutputFormat")
        '''
        return rdd.foreach(f)
    
    
    