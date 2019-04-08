

import KProto.toTurple
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.clustering4ever.clustering.kcenters.rdd.KPrototypes
import org.clustering4ever.math.distances.mixt.HammingAndEuclidean
import org.clustering4ever.util.ScalaCollectionImplicits //Create a SparkContext to initialize Spark
val conf = new SparkConf()
conf.setMaster("local")
conf.setAppName("K-Proto poc")
val sc = new SparkContext(conf)

// Load the text into a Spark RDD, which is a distributed representation of each line of text
val textFile = sc.textFile("/Users/alexander.yushchenko/project/kproto2/src/main/resources/data.txt")
val seqs = textFile.map(f => toTurple(f, 9)).collect()
val kprototypes = KPrototypes(4, HammingAndEuclidean[Seq[Int], Seq[Double]](), 0.1, 1000, StorageLevel.MEMORY_ONLY)
val cs = ScalaCollectionImplicits.mixtToClusterizable(seqs.toSeq)
val result = kprototypes.fit(sc.parallelize(cs))
