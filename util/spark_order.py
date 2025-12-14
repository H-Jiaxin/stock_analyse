from pyspark.sql import SparkSession

# 启动 Spark Session
spark = SparkSession.builder \
    .appName("StockDataAnalyse") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
