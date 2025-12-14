from util import spiderTool, feature_ng, dataClean, visualize

if __name__ == '__main__':
    data_path = "./data/kechuangban_kline_data.csv"
    cleaned_data = './data/cleaned_stock_data'
    fn_data = './data/final_high_vol_features'
    # 选择股票范围，默认采用市盈率最高的科创版股票数据（更多股票获取请查看原函数）
    stock_codes = spiderTool.get_stock_code_list(page_size=80)

    # 数据获取
    spiderTool.get_stock_data(data_path, stock_codes)

    # 数据加载与清洗
    dataClean.clean_stock_data(data_path, output=cleaned_data)

    # 特征工程
    feature_ng.feature_engneer(cleaned_data, 100, final_output_path=fn_data)

    # 可视化
    visualize.visualize_results(fn_data)

    



    