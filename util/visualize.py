import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .spark_order import spark
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pyspark.sql.functions import col

import matplotlib.font_manager as fm

def apply_system_font():
    # 1. 自动寻找系统里包含 'CJK', 'Hei', 'SimSun' 等关键词的字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    zh_fonts = [f for f in all_fonts if any(kw in f for kw in ['CJK', 'Hei', 'SimSun', 'STHeiti', 'Micro Hei'])]
    print(f"系统中找到的中文字体: {zh_fonts}")
    if zh_fonts:
        # 找到什么就用什么，不再硬编码 Noto Sans CJK TC
        plt.rcParams['font.sans-serif'] = [zh_fonts[0]] + plt.rcParams['font.sans-serif']
        print(f"自动选择系统字体: {zh_fonts[0]}")
    else:
        print("警告：系统内未找到任何中文字体，请先执行 sudo apt install fonts-noto-cjk")

    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def eda_visualization(raw_data_path, save_path="./data/output/"):
    # 1. 环境初始化与样式设置
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    sns.set_theme(style="whitegrid", font='Noto Serif CJK JP', rc={'axes.unicode_minus': False})
    # 定义专业配色方案
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 加载数据
    print("加载数据...")
    spark_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)
    
    # ---------------------------------------------------------
    # 图 1: 缺失值模式热力图 (修复日期显示)
    # ---------------------------------------------------------
    print("正在生成缺失值热力图...")
    pdf = spark_df.select("Date", "Code", "Close").toPandas()
    pdf['Date'] = pd.to_datetime(pdf['Date'])
    
    # 排序逻辑保持不变
    df_pivot = pdf.pivot(index='Code', columns='Date', values='Close')
    first_valid_date = df_pivot.apply(lambda x: x.first_valid_index(), axis=1)
    df_pivot_sorted = df_pivot.loc[first_valid_date.sort_values().index]

    plt.figure(figsize=(16, 9), dpi=200)
    all_dates = df_pivot_sorted.columns
    n_dates = len(all_dates)
    
    # 设定日期数量
    n_ticks = 30 
    tick_indices = [i for i in range(0, n_dates, n_dates // n_ticks)]
    tick_labels = [all_dates[i].strftime('%Y-%m') for i in tick_indices]

    ax = sns.heatmap(
        df_pivot_sorted.isnull(), 
        cbar=False, 
        cmap="coolwarm", 
        yticklabels=False, 
        xticklabels=False  # 先关闭自动标签
    )
    # 手动添加
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.title("科创板数据完整度分布图 (按上市时间排序)", fontsize=16, pad=1)
    plt.xlabel("交易日期 (年-月)", fontsize=12)
    plt.ylabel("不同股票样本 (垂直轴)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "EDA_Missing_Heatmap.png"))

    # ---------------------------------------------------------
    # 图 2: 股价走势对比 (优化：归一化处理 Base 100)
    # ---------------------------------------------------------
    print("正在生成归一化股价对比图...")
    # 选取波动明显的 5 支股票
    target_stocks = pdf.groupby('Code')['Close'].std().nlargest(5).index.tolist()
    subset_df = pdf[pdf['Code'].isin(target_stocks)].copy()

    # 关键优化：归一化 (Normalization)
    # 解决绝对价格差异大（如 10元 vs 500元）无法对比的问题
    # 将每支股票的初始价格设为 100，观察其相对涨跌幅
    def rebase_price(group):
        first_price = group.sort_values('Date')['Close'].iloc[0]
        group['Rebased_Close'] = (group['Close'] / first_price) * 100
        return group

    subset_df = subset_df.groupby('Code', group_keys=False).apply(rebase_price)

    plt.figure(figsize=(12, 6), dpi=200)
    sns.lineplot(data=subset_df, x="Date", y="Rebased_Close", hue="Code", palette=colors, linewidth=2)
    
    # 添加基准线
    plt.axhline(100, color='black', linestyle='--', alpha=0.5)
    plt.title("代表性股票相对涨跌幅趋势", fontsize=16)
    plt.ylabel("归一化价格 (初始=100)", fontsize=12)
    plt.xlabel("日期", fontsize=12)
    plt.legend(title="股票代码", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "EDA_Price_Normalized.png"))

    # ---------------------------------------------------------
    # 图 3: 成交量分布 (优化：KDE 曲线 & 统计细节)
    # ---------------------------------------------------------
    print("正在生成成交量分布图...")
    vol_data = spark_df.select("Volume").toPandas()
    
    plt.figure(figsize=(10, 6), dpi=200)
    # 优化：结合直方图与核密度估计曲线 (KDE)
    sns.histplot(vol_data['Volume'], bins=60, kde=True, color="#2c3e50", log_scale=True, edgecolor='w')
    
    plt.title("成交量分布特征统计 (对数坐标)", fontsize=16)
    plt.xlabel("成交量 (Volume)", fontsize=12)
    plt.ylabel("频率分布 (Density)", fontsize=12)
    
    # 在图中添加中位数标注
    median_vol = vol_data['Volume'].median()
    plt.axvline(median_vol, color='red', linestyle='-', label=f'中位数: {median_vol:.0f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "EDA_Volume_Distribution.png"))

    print(f"原始数据可视化完成，已保存至: {save_path}")

def visualize_results(final_parquet_path, diagram_path = './data/diagram/', tscode = None, spark_session = spark):
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
    
    if tscode is not None:
        # 如果指定了特定股票代码，则绘制该股票的价格与 MA 趋势图
        stock_code = tscode
    else:
        # 否则默认选取波动率最高的股票
        stock_code = avg_vol_df.iloc[0]['ts_code']
    df_single = df_pandas[df_pandas['ts_code'] == stock_code].sort_values(by='trade_date')
    
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
        title=f'{stock_code} 股价与多周期移动平均线 (MA) 趋势分析',
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
    raw_data = "./data/kechuangban_kline_data.csv"
    # 绘图前调用
    # apply_system_font()
    # eda_visualization(raw_data)
    visualize_results(final_data)