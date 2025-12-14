import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lag, stddev, desc
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType
from spark_order import spark

# 假设 Spark Session 已经在主程序中初始化

def feature_engneer(cleaned_data_path, target_count=10, final_output_path="./data/final_high_vol_features"):
    """
    加载清洗后的数据，计算多周期特征，并筛选出平均波动率最高的 N 支股票。

    :param cleaned_data_path: 清洗后数据（Parquet格式）的路径。
    :param target_count: 最终用于分析的股票数量（按平均波动率降序）。
    :param final_output_path: 最终特征数据集的保存路径。
    :return: 包含所有特征和目标筛选后的 PySpark DataFrame。
    """
    
    print("=== 开始特征工程与高波动性筛选 ===")
    
    df_history = spark.read.parquet(cleaned_data_path)
    window_spec = Window.partitionBy("ts_code").orderBy("trade_date")
    df_features = df_history

    # 收益率 (Daily_Return) 和波动率 (Vol_20D)
    # 计算 Daily_Return (用于计算波动率)
    df_features = df_features.withColumn(
        "prev_close", lag(col("close"), 1).over(window_spec)
    )
    df_features = df_features.withColumn(
        "Daily_Return", (col("close") - col("prev_close")) / col("prev_close")
    )
    
    # 计算 Vol_20D (20日收益率标准差，短期波动率)
    window_spec_stddev = window_spec.rowsBetween(-19, 0)
    df_features = df_features.withColumn(
        "Vol_20D",
        stddev(col("Daily_Return")).over(window_spec_stddev)
    )
    
    # B. 价格移动平均线 (MA) - 短期、中期、中长期、长期
    window_spec_ma5 = window_spec.rowsBetween(-4, 0)  # 5日
    window_spec_ma20 = window_spec.rowsBetween(-19, 0) # 20日
    window_spec_ma60 = window_spec.rowsBetween(-59, 0) # 60日 (中长期特征)
    window_spec_ma120 = window_spec.rowsBetween(-119, 0) # 120日 (长期特征)
    
    df_features = df_features.withColumn("MA5", avg(col("close")).over(window_spec_ma5))
    df_features = df_features.withColumn("MA20", avg(col("close")).over(window_spec_ma20))
    df_features = df_features.withColumn("MA60", avg(col("close")).over(window_spec_ma60))
    df_features = df_features.withColumn("MA120", avg(col("close")).over(window_spec_ma120))
    
    # 量能移动平均线 (VMA5)
    df_features = df_features.withColumn(
        "VMA5",
        avg(col("volume")).over(window_spec_ma5)
    )
    
    df_analyzable = df_features.na.drop(subset=["Vol_20D", "MA20"])
    
    # 计算每只股票在整个分析周期内的平均波动率 (Avg_Volatility)
    df_avg_vol = df_analyzable.groupBy("ts_code").agg(
        avg("Vol_20D").alias("Avg_Volatility")
    )
    
    # 按平均波动率降序排序，选取目标数量
    target_vol_codes = df_avg_vol.orderBy(desc("Avg_Volatility")) \
        .limit(target_count) \
        .select("ts_code") \
        .rdd.flatMap(lambda x: x) \
        .collect()
        
    print(f"according Avg_Volatility filter from height tp low，{len(target_vol_codes)} in total")

    # 4. 过滤最终数据集，只保留高波动性股票的历史数据
    final_analysis_df = df_analyzable.filter(col("ts_code").isin(target_vol_codes))
    final_analysis_df = final_analysis_df.drop("prev_close")
    
    final_analysis_df.write.parquet(final_output_path, mode="overwrite")
    print(f"\n--- complete ---")
    print(f"final data line: {final_analysis_df.count()}")
    print(f"save to: {final_output_path}")
    
    return final_analysis_df

if __name__ == '__main__':
    data = './data/cleaned_stock_data'
    analysis_df = feature_engneer(data, 100)
    print("scheme detail:")
    analysis_df.printSchema()
    spark.stop()
