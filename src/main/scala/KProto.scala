import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.clustering4ever.clustering.kcenters.rdd.{KPrototypes, KPrototypesModels}
import org.clustering4ever.math.distances.mixt.HammingAndEuclidean
import org.clustering4ever.util.ScalaCollectionImplicits

object KProto {

  def main(args: Array[String]) {

    //Create a SparkContext to initialize Spark
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("K-Proto poc")
    val sc = new SparkContext(conf)

    // Load the text into a Spark RDD, which is a distributed representation of each line of text
    val textFile = sc.textFile("src/main/resources/data.txt")
    val seqs = textFile.map(f => toTurple(f, 9)).collect()
    println("Hi there")
    println(seqs)
    val kprototypes = KPrototypes(4, HammingAndEuclidean[Seq[Int], Seq[Double]](), 0.1, 1000, StorageLevel.MEMORY_ONLY)


    val cs = ScalaCollectionImplicits.mixtToClusterizable(seqs.toSeq)

    val result: KPrototypesModels[Seq[Int], Seq[Double], HammingAndEuclidean] = kprototypes.fit(sc.parallelize(cs))
  }

  def toTurple(line: String, numericIndex: Int): (Seq[Int], Seq[Double]) = {
    val values: Array[String] = line.split("[ ]+")
    val numeric: Seq[Double] = values.take(numericIndex).map(x => x.toDouble)
    val categorical: Seq[Int] = values.takeRight(values.length - numericIndex).map(x => x.toInt)
    (categorical, numeric)
  }
}