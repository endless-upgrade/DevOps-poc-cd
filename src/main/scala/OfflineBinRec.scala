import java.io.File

import com.typesafe.config._
import it.reply.data.devops.{BinaryALS, BinaryALSValidator}
import it.reply.data.pasquali.Storage
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession

import scala.reflect.io.Path
import scala.util.Try

object OfflineBinRec {

  var mr : BinaryALS = null
  var spark : SparkSession = null

  def main(args: Array[String]): Unit = {

    var toggles : Config = null

    if(args.length != 0)
      toggles = ConfigFactory.parseFile(new File(args(0)))
    else
      toggles = ConfigFactory.load

    mr = BinaryALS().initSpark("test", "local")

    spark = mr.spark

    val ratings = spark.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("data/ratings.csv")
      .drop("time")
      .rdd.map { rate =>
      Rating(rate(0).toString.toInt, rate(1).toString.toInt, rate(2).toString.toDouble)
    }.cache()

    val Array(train, test) = ratings.randomSplit(Array(0.8, 0.2))

    mr.trainModelBinary(train, 10, 10, 0.1)

    print("Model Trained")

    if(toggles.getBoolean("toggle.evaluate.evaluate")){

      val validator = BinaryALSValidator(mr.model).init(test)

      if(toggles.getBoolean("toggle.evaluate.accuracy"))
        println(s"accuracy = ${(validator.accuracy*100).toInt}%")

      if(toggles.getBoolean("toggle.evaluate.precision"))
        println(s"precision = ${(validator.precision*100).toInt}%")

      if(toggles.getBoolean("toggle.evaluate.recall"))
        println(s"recall = ${(validator.recall*100).toInt}%")

    }


    if(toggles.getBoolean("toggle.store.store")){

      val path = toggles.getString("toggle.store.path")
      val zipIt = toggles.getBoolean("toggle.store.zip")

      storeModel(path, zipIt)
    }
  }


  def storeModel(path : String, zip : Boolean = false) : Unit = {

    mr.storeModel(path+"/poc")

    if(zip){
      val storage = Storage()
      storage.zipModel(path+"/poc", path+"/poc.zip")

      Try(Path(path+"/poc").deleteRecursively())
    }
  }

}
