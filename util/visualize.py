import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .spark_order import spark
import os

def visualize_results(final_parquet_path, diagram_path = './data/diagram/', spark_session = spark):
    """
    加载最终特征数据，生成关键可视化图表，并保存为HTML文件。
    
    :param final_parquet_path: 最终特征数据集路径。
    :param spark_session: 活跃的 SparkSession。
    """
    
    print("\n=== 开始结果可视化 ===")
    if not os.path.exists(diagram_path):
        os.makedirs(diagram_path, exist_ok=True) 
    
    # 1. 从 Parquet 加载数据并转换为 Pandas DataFrame
    try:
        df_spark = spark_session.read.parquet(final_parquet_path)
        df_pandas = df_spark.toPandas()
    except Exception as e:
        print(f"无法加载最终数据集: {e}")
        print("请确保特征工程步骤已成功运行并生成 Parquet 文件。")
        return

    # 确保日期是 Pandas 的 datetime 类型
    df_pandas['trade_date'] = pd.to_datetime(df_pandas['trade_date'])
    
    # 计算每只股票的整体平均波动率
    avg_vol_df = df_pandas.groupby('ts_code')['Vol_20D'].mean().reset_index()
    avg_vol_df.columns = ['ts_code', 'Avg_Volatility']
    
    # 按波动率降序排序 (与筛选逻辑一致)
    avg_vol_df = avg_vol_df.sort_values(by='Avg_Volatility', ascending=False)
    
    # 使用 Plotly 生成交互式条形图
    fig1 = px.bar(
        avg_vol_df, 
        x='ts_code', 
        y='Avg_Volatility', 
        title='目标股票池平均波动率 (Vol_20D) 对比'
    )
    fig1.update_xaxes(title_text='股票代码')
    fig1.update_yaxes(title_text='平均 20 日波动率')
    
    # 保存为 HTML 文件
    html_path_1 = diagram_path + "viz_1_avg_volatility_comparison.html"
    fig1.write_html(html_path_1)
    print(f"图表1 (波动率对比) 已保存至: {html_path_1}")
    
    # 选取波动率最高的股票进行详细分析
    top_stock_code = avg_vol_df.iloc[0]['ts_code']
    df_single = df_pandas[df_pandas['ts_code'] == top_stock_code].sort_values(by='trade_date')
    
    # 创建 Plotly 趋势图
    fig2 = go.Figure()
    
    # 添加收盘价
    fig2.add_trace(go.Scatter(x=df_single['trade_date'], y=df_single['close'], mode='lines', name='收盘价 (Close)', line=dict(color='black')))
    # 添加短期 MA (MA5)
    fig2.add_trace(go.Scatter(x=df_single['trade_date'], y=df_single['MA5'], mode='lines', name='MA5 (短期)', line=dict(color='blue', dash='dot')))
    # 添加中期 MA (MA20)
    fig2.add_trace(go.Scatter(x=df_single['trade_date'], y=df_single['MA20'], mode='lines', name='MA20 (中期)', line=dict(color='orange')))
    # 添加长期 MA (MA60)
    fig2.add_trace(go.Scatter(x=df_single['trade_date'], y=df_single['MA60'], mode='lines', name='MA60 (中长期)', line=dict(color='red')))

    fig2.update_layout(
        title=f'{top_stock_code} 股价与多周期移动平均线 (MA) 趋势分析',
        xaxis_title='交易日期',
        yaxis_title='价格 (元)',
        xaxis_rangeslider_visible=True # 允许在图表底部滑动缩放，增强交互性
    )
    
    # 保存为 HTML 文件
    html_path_2 = diagram_path + "viz_2_top_stock_price_ma_trend.html"
    fig2.write_html(html_path_2)
    print(f"图表2 (趋势分析) 已保存至: {html_path_2}")
    

    # --- 可视化 3: 日收益率分布 (Histogram) ---
    
    # 分析所有目标股票的日收益率分布
    fig3 = px.histogram(
        df_pandas, 
        x="Daily_Return", 
        nbins=100, # 使用100个直方图桶，使分布更细致
        title='目标股票池日收益率分布 (Daily_Return)'
    )
    fig3.update_layout(xaxis_title='日收益率 (%)', yaxis_title='频数')
    
    html_path_3 = diagram_path + "viz_3_daily_return_histogram.html"
    fig3.write_html(html_path_3)
    print(f"图表3 (收益率分布) 已保存至: {html_path_3}")

    spark.stop()

if __name__ == '__main__':
    final_data = "./data/final_high_vol_features"
    visualize_results(final_data)