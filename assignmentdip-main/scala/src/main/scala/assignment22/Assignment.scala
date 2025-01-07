package assignment22

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array, asc, col, desc, explode, first, max, mean, min, second, sum, udf, when, year}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import breeze.plot._
import scala.Console.println
import org.apache.spark.sql.functions.not


class Assignment {

  val spark: SparkSession = SparkSession.builder()
    .appName("AssignmentScala")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")
  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read.option("inferSchema","true").option("header","true").csv("data/dataD2.csv")

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read.option("inferSchema","true").option("header","true").csv("data/dataD3.csv")

  val dirtyData: DataFrame = spark.read.schema("a Double,b Double,LABEL String").option("header", "true").option("nullvalue","null")
    .option("mode","DROPMALFORMED").csv("data/dataD2_dirty.csv")

  dirtyData.na.drop("all")
  dirtyData.printSchema()
  dirtyData.show(20)

  val df = dirtyData.where(!col("LABEL").contains("Unknown"))
  val dirtyDataWithLabels: DataFrame = df.withColumn("c", labelToNumeric(df("LABEL")))

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  def labelToNumeric: UserDefinedFunction = {
    udf((LABEL: String) => {
      LABEL match {
        case "Fatal" => 0.0
        case "Ok" => 1.0
      }
    })
  }
  val dataD2WithLabels: DataFrame = dataD2.withColumn("c", labelToNumeric(dataD2("LABEL")))// REPLACE with actual implementation



  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    //Array.empty  // REPLACE with actual implementation
    val min_max_a = df.agg(min("a"), max("a")).head()
    val a_max = min_max_a.getDouble(1)
    val a_min = min_max_a.getDouble(0)
    val min_max_b = df.agg(min("b"), max("b")).head()
    val b_max = min_max_b.getDouble(1)
    val b_min = min_max_b.getDouble(0)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b")).setHandleInvalid("skip")
      .setOutputCol("features")
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    // Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    //Scale data using Min Max scaler
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(transformedData)
    val scaledData = scalerModel.transform(transformedData)
    val kmeans = new KMeans()
      .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    kmModel.summary.predictions.show()
    val clusters: Array[(Double,Double)] = kmModel.clusterCenters.map(x=>(x(0),x(1)))
    // Rescale Data back to original scale
    val rescaledResult = rescaleDataBackToOriginal2D(a_max,a_min,b_max,b_min,clusters)
    rescaledResult.foreach(println)
    rescaledResult
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    //Array.empty  // REPLACE with actual implementation
    val min_max_a = df.agg(min("a"), max("a")).head()
    val a_max = min_max_a.getDouble(1)
    val a_min = min_max_a.getDouble(0)
    val min_max_b = df.agg(min("b"), max("b")).head()
    val b_max = min_max_b.getDouble(1)
    val b_min = min_max_b.getDouble(0)
    val min_max_c = df.agg(min("c"), max("c")).head()
    val c_max = min_max_c.getDouble(1)
    val c_min = min_max_c.getDouble(0)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    // Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    // Scale data using Min Max scaler
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(transformedData)
    val scaledData = scalerModel.transform(transformedData)
    val kmeans = new KMeans()
      .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    kmModel.summary.predictions.show()
    val clusters: Array[(Double,Double, Double)] = kmModel.clusterCenters.map(x=>(x(0),x(1),x(2)))
    kmModel.clusterCenters.foreach(println)
    // Rescale Data back to original scale
    val rescaledResult = rescaleDataBackToOriginal3D(a_max,a_min,b_max,b_min,c_max,c_min,clusters)
    rescaledResult.foreach(println)
    rescaledResult
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    //Array.empty  // REPLACE with actual implementation
    val min_max_a = df.agg(min("a"), max("a")).head()
    val a_max = min_max_a.getDouble(1)
    val a_min = min_max_a.getDouble(0)
    val min_max_b = df.agg(min("b"), max("b")).head()
    val b_max = min_max_b.getDouble(1)
    val b_min = min_max_b.getDouble(0)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c")).setHandleInvalid("skip")
      .setOutputCol("features")
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    // Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    // Scale data using Min Max scaler
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(transformedData)
    val scaledData = scalerModel.transform(transformedData)
    val kmeans = new KMeans()
      .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    val a:DataFrame=kmModel.summary.predictions
    //a.show()
    val b= a.filter("LABEL=='Fatal'")
    //b.show()
    val c= b.groupBy("prediction").count().sort(col("count").desc)
    //c.show()
    val d=c.drop("count")
    //d.show()
    val e=d.take(2)
    //e.foreach(println)
    val firstVal:Int=e(0).getInt(0)
    println(firstVal)
    val secondVal=e(1).getInt(0)
    println(secondVal)
    val clusters = kmModel.clusterCenters.map(x=>(x(0),x(1)))
    // converting to list so as to create Array
    val ls=List(clusters(firstVal),clusters(secondVal))
    val topTwoClusters:Array[(Double, Double)]= ls.toArray
    topTwoClusters.foreach(println)
    //Rescale Data back to original scale
    val rescaledResult = rescaleDataBackToOriginal2D(a_max,a_min,b_max,b_min,topTwoClusters)
    rescaledResult.foreach(println)
    rescaledResult
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    // Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    // Scale data using Min Max scaler
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(transformedData)
    val scaledData = scalerModel.transform(transformedData)
    val result = new Array[(Int,Double)](high-low+1)
    val kscore = new Array[Double](high-low+1)
    val sil = new Array[Double](high-low+1)
    var i: Int = 0
    for(k:Int <- low to high) {
      val kmeans = new KMeans()
        .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
      val kmModel = kmeans.fit(scaledData)
      //kmModel.summary.predictions.show()

      val predictions = kmModel.transform(scaledData)
      val evaluator = new ClusteringEvaluator().setFeaturesCol("scaledFeatures")
      val silhouetteScore:Double = evaluator.evaluate(predictions)
      //println(silhouetteScore)
      result(i) = (k,silhouetteScore)
      kscore(i) = (k)
      sil(i) = (silhouetteScore)
      i = i + 1
    }
    //ans.foreach(println)
    plotSIlhouette(sil,kscore)
    result
  }

  def plotSIlhouette(ss:Array[Double],k:Array[Double]): Unit ={
    val fig = Figure()
    val p = fig.subplot(0)
    //val x :Array[Double]= {}
    p+=plot(k,ss)
    p.title = "Silhouette Visualization"
    p.xlabel = "K"
    p.ylabel = "Silhouette score"
    fig.saveas("silhouettescore.png")
  }

  def rescaleDataBackToOriginal2D(amax:Double, amin:Double, bmax:Double, bmin:Double, clusters:Array[(Double,Double)]): Array[(Double, Double)] ={
    val res: Array[(Double,Double)] = clusters.map(x => ((amin + (amax - amin) * x._1),(bmin + (bmax - bmin) * x._2)))
    res
  }

  def rescaleDataBackToOriginal3D(amax:Double, amin:Double, bmax:Double, bmin:Double, cmax:Double, cmin:Double, clusters:Array[(Double,Double,Double)]): Array[(Double, Double,Double)] ={
    val res: Array[(Double,Double,Double)] = clusters.map(x => ((amin + (amax - amin) * x._1), (bmin + (bmax - bmin) * x._2), (cmin + (cmax - cmin) * x._3 )))
    res
  }

}
