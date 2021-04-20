package com.example.apps

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.math.random

object Pi {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Pi").master("local").getOrCreate()
    val partitions = if (args.length > 0) args(0).toInt else 2
    val pi = calculatePi(spark.sparkContext, partitions)
    println(s"pi = $pi")
    spark.stop()
  }

  def calculatePi(sc:SparkContext, partitions:Int): Double = {
    val n = (100000 * partitions).toInt

    def f(i: Int): Int = {
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x*x + y*y <= 1) 1 else 0
    }

    // P(x^2+y^2<=1 | -1<=x<=0, -1<=y<=0) = pi/4 = count/n
    val count = sc.parallelize(1 until n, partitions).map(f).reduce(_ + _)
    4.0 * count / n
  }
}
