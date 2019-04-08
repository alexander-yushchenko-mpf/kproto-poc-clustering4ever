import org.apache.commons.math3.linear.RealVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.clustering4ever.clustering.kcenters.rdd.{KMeans, KMeansModel, KPrototypes}
import org.clustering4ever.clusterizables.EasyClusterizable
import org.clustering4ever.math.distances.mixt.HammingAndEuclidean
import org.clustering4ever.util.ScalaCollectionImplicits

import scala.collection.{GenSeq, immutable, mutable}
import org.clustering4ever.math.distances.scalar.Euclidean
import org.clustering4ever.vectorizables.Vectorizable
import org.clustering4ever.vectors.{GVector, ScalarVector}

object Kmean {

  def main(args: Array[String]) {

    //Create a SparkContext to initialize Spark
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("K-Proto poc")
    val sc = new SparkContext(conf)

    val datasetSize = 500000
    val dim = 4
    val useAggregationDS = true
    val dp = 16

    val path = "/tmp/aggregation.csv"
    val rawRdd: RDD[Seq[Double]] = sc.textFile(path, 6).map(x => Seq(x.split(",").map(_.toDouble):_*))

//    val rdd: RDD[EasyClusterizable[Vectorizable[Seq[Double]], ScalarVector[Seq[Double]]]] = rawRdd.zipWithIndex.map{ case (v, id) =>  EasyClusterizable(id, Vectorizable[Seq[Double]](v), ScalarVector(v)) }.cache
//    val labelsPath = "/tmp/labels"
//
////    val path = "/tmp/aggregation.csv"
////    val rawRdd = if( useAggregationDS ) sc.textFile(path, dp).map( x => Seq(x.split(",").map(_.toDouble):_*)) else sc.parallelize(List.fill(datasetSize)(mutable.ArrayBuffer.fill(dim)(scala.util.Random.nextDouble)), dp)
////    val seq = ScalaCollectionImplicits.scalarToClusterizable(Seq(rawRdd.collect().toSeq))
////    val rdd = sc.parallelize(seq)
////
////    val labelsPath = "/tmp/labels"
//    rdd.foreach(println)

//
    val k = 7
    val iterMax = 40
    val epsilon = 0.0
    val metric: Euclidean[Seq[Double]] = Euclidean[Seq[Double]](false)

    val rddPredict: RDD[ScalarVector[Seq[Double]]] = rawRdd.map(x => ScalarVector(x))


    val t1 = System.currentTimeMillis
    val model: KMeansModel[Seq[Double], Euclidean] = KMeans.fit(rawRdd, k, metric, epsilon, iterMax, StorageLevel.MEMORY_ONLY)
//    val res: model.ClusterID = model.centerPredict(ScalarVector(Seq(8.6, 25.65)))
//    val res: model.ClusterID = model.centerPredict(ScalarVector(Seq(6.2, 4.25)))
    val res = model.centerPredict(rddPredict)

    res.foreach(println)
    model.centers.foreach(println)
    val t2 = System.currentTimeMillis
    (t2 - t1) / 1000D
  }
}