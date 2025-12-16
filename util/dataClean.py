import sys
from pyspark.sql.functions import col, sum, when, last
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, DateType, StringType
from .spark_order import spark

# 超过阈值说明偏差较大，数据可信度不高
PRICE_FILL_THRESHOLD_PCT = 5.0  
VOLUME_FILL_THRESHOLD_PCT = 1.0 

cols_mapping = {
    "Code": ("ts_code", StringType()),
    "Date": ("trade_date", DateType()),
    "Open": ("open", DoubleType()),
    "Close": ("close", DoubleType()),
    "High": ("high", DoubleType()),
    "Low": ("low", DoubleType()),
    "Volume": ("volume", DoubleType()),
    "Turnover": ("turnover", DoubleType()),
    "Amplitude": ("amplitude", DoubleType()), 
    "Change_Pct": ("change_pct", DoubleType()),
    "Change_Amt": ("change_amt", DoubleType()),
    "Turnover_Rate": ("turnover_rate", DoubleType())
}
cols_to_check = [
    'open', 'close', 'high', 'low', 
    'volume', 'turnover', 'change_pct', 'turnover_rate'
]

def clean_stock_data(file_path, output = "./data/cleaned_stock_data"):
    spark_df = spark.read.csv(file_path, header=True, inferSchema=False) 
    df_cleaned = spark_df
    for original_name, (new_name, new_type) in cols_mapping.items():
        df_cleaned = df_cleaned.withColumn(new_name,col(original_name).cast(new_type))
    final_cols = [v[0] for v in cols_mapping.values()]
    df_cleaned = df_cleaned.select(*final_cols)
    print("=== 数据加载完成，开始进行缺失值诊断 ===")
    total_rows = df_cleaned.count()

    missing_count_expressions = [
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(f"{c}_missing_count")
        for c in cols_to_check
    ]

    missing_counts_df = df_cleaned.agg(*missing_count_expressions)
    missing_counts_pd = missing_counts_df.toPandas()

    for c in cols_to_check:
        count_col = f"{c}_missing_count"
        percent_col = f"{c}_missing_percent"
        missing_counts_pd[percent_col] = (
            missing_counts_pd[count_col] / total_rows * 100
        ).round(2)

    print(f"总记录数：{total_rows}")
    print("\n--- 缺失值诊断报告 ---")
    print(missing_counts_pd[[f"{c}_missing_count" for c in cols_to_check] + [f"{c}_missing_percent" for c in cols_to_check]])
    missing_counts_pd.to_csv("./data/missing_value_report.csv", index=False)
    print("缺失值诊断报告已保存至 ./data/missing_value_report.csv")

    # 过滤无效数据
    df_filtered = df_cleaned.filter(
        (col("close").isNotNull()) & (col("close") > 0) & (col("volume") >= 0)
    )

    df_current = df_filtered
    
    # 交易量/额的 0 值填充
    volume_pct = missing_counts_pd.loc[0, 'volume_missing_percent']
    turnover_pct = missing_counts_pd.loc[0, 'turnover_missing_percent']
    
    if volume_pct <= VOLUME_FILL_THRESHOLD_PCT and turnover_pct <= VOLUME_FILL_THRESHOLD_PCT:
        df_current = df_current.na.fill(0.0, subset=['volume', 'turnover'])
        print(f"交易量/额缺失率 ({volume_pct}%) 低于 {VOLUME_FILL_THRESHOLD_PCT}%，已执行 0 填充。")
    else:
        print(f"交易量/额缺失率 ({volume_pct}%) 过高，跳过 0 填充。")


    # 前向填充
    window_spec = Window.partitionBy("ts_code").orderBy("trade_date")
    price_cols = ['open', 'close', 'high', 'low']
    df_final = df_current
    
    for price_col in price_cols:
        price_pct = missing_counts_pd.loc[0, f'{price_col}_missing_percent']
        
        if price_pct <= PRICE_FILL_THRESHOLD_PCT:
            df_final = df_final.withColumn(
                price_col,
                last(price_col, ignorenulls=True).over(window_spec.rowsBetween(-sys.maxsize, 0))
            )
            print(f"{price_col} 缺失率 ({price_pct}%) 低于 {PRICE_FILL_THRESHOLD_PCT}%，已执行前向填充。")
        else:
            print(f"{price_col} 缺失率 ({price_pct}%) 过高，跳过前向填充。")

    final_cleaned_df = df_final.select(
        "ts_code", "trade_date", 
        *price_cols, 
        "volume", "turnover", 
        "change_pct", "turnover_rate", 
        "Amplitude", "Change_Amt"
    )
    print("\n--- 数据清洗完成 ---")
    
    final_cleaned_df.write.parquet(output, mode="overwrite")
    print(f"清洗后的数据已保存至: {output} 清洗后的数据行数: {final_cleaned_df.count()}")

if __name__ == "__main__":
    # 示例用法
    data_path = "./data/kechuangban_kline_data.csv"
    clean_stock_data(data_path)
    spark.stop()