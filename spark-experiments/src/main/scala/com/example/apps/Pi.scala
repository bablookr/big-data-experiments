package com.example.apps

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.math.random

object Pi {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Pi").master("local").getOrCreate()
    val partitions = if (args.length > 0) args(0).toInt else 10
    val pi = calculatePi(spark.sparkContext, partitions)
    println(s"pi = $pi")
    spark.stop()
  }

  def calculatePi(sc:SparkContext, partitions:Int): Double = {
    val n = 100000L * partitions

    def f(i: Long): Int = {
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x*x + y*y <= 1) 1 else 0
    }

    // Number of points (at random) inside a quarter of circle with radius 1 unit
    val count = sc.parallelize(1 until n, partitions).map(f).reduce(_ + _)
    4.0 * count / n
  }
}
