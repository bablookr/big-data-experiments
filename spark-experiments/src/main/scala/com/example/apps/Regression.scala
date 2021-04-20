package com.example.apps

import java.util.Random
import scala.math.exp
import org.apache.spark.sql.SparkSession
import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.SparkContext

object Regression {
  val n = 10000
  val dim = 10;
  val seed = 42
  val iterations = 5

  case class DataPoint(x:Vector[Double], y:Double)

  def generatedData: Array[DataPoint] = {
    def generatePoint(i:Int): DataPoint = {
      val y = if (i % 2 == 0) -1 else 1
      val x = DenseVector.fill(dim) {0.7 * y + new Random(seed).nextGaussian()}
      DataPoint(x, y)
    }
    Array.tabulate(n)(generatePoint)
  }

  def initializeWeights: DenseVector[Double] = {
    DenseVector.fill(dim){2 * new Random(seed).nextDouble() - 1}
  }

  def trainModel(sc: SparkContext, partitions: Int): Unit = {
    val points = sc.parallelize(generatedData, partitions)
    val w = initializeWeights
    println(s"Iteration 0 : w = $w")
    for(i <- 1 to iterations) {
      val gradient = points.map(p => p.x * (1 / (1 + exp(-p.y * (w.dot(p.x)))) - 1) * p.y).reduce(_ + _)
      w -= gradient
      print(s"Iteration $i : w = $w")
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Regression").master("local").getOrCreate()
    val partitions = if (args.length > 0) args(0).toInt else 10
    trainModel(spark.sparkContext, partitions)
    spark.stop()
  }
}
