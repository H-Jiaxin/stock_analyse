from util import feature_ng, dataClean, visualize
from util.spiderTool import spiderTool

if __name__ == '__main__':
    data_path = "./data/kechuangban_kline_data.csv"
    cleaned_data = './data/cleaned_stock_data'
    fn_data = './data/final_high_vol_features'

    # 选择股票范围，默认采用市盈率最高的科创版股票数据（更多股票获取请查看原函数）
    stock_codes = spiderTool.get_tscode_list(200)
    # 数据获取
    spiderTool.get_stock_data_concurrent(data_path, stock_codes)
    # spiderTool.get_stock_data(data_path, stock_codes)

    # 数据展示
    visualize.eda_visualization(data_path)

    # 数据清洗
    dataClean.clean_stock_data(data_path, output=cleaned_data)

    # 特征工程
    feature_ng.feature_engneer(cleaned_data, 50, final_output_path=fn_data)

    # 可视化
    visualize.visualize_results(fn_data)

    



    