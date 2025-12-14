import requests
import pandas as pd
import json
import time
import random

def generate_random_headers(url):
        user_agent_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) Gecko/20100101 Firefox/98.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/99.0.1150.36',
        ]

        return {
            'User-Agent': random.choice(user_agent_list),
            'content-type': 'application/javascript; charset=UTF-8',
            'Connection': 'keep-alive',
            'Referer': url
        }


class spiderTool:
    def get_stock_kline_data(stock_code, market_code, limit):
        """
        使用东方财富历史K线接口获取股票数据。

        :param stock_code: 股票代码 (如 '300001')
        :param market_code: 市场代码 (深圳为 '0', 上海为 '1')
        :param limit: 获取的数据条数 (例如 1500 条日K线约等于 5 年数据)
        :return: 包含股票历史数据的 Pandas DataFrame
        """
        # 构造 secid
        
        secid = f"{market_code}.{stock_code}"
        base_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        
        params = {
            'secid': secid,
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': 101,  # 日K线
            'fqt': 1,
            'end': '20250101', # 结束日期
            'lmt': limit
        }
        headers = generate_random_headers(f"https://quote.eastmoney.com/kcb/{stock_code}.html")
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status() # 检查HTTP请求是否成功
            # 处理掉回调函数名 格式是：jsonp_XX({...})
            jsonp_text = response.text
            start_index = jsonp_text.find('{')
            end_index = jsonp_text.rfind('}') + 1
            json_data = jsonp_text[start_index:end_index]
            data = json.loads(json_data)
            code = data['data']['code']
            klines = data['data']['klines']
            
            records = []
            for line in klines:
                records.append([code] + line.split(','))
                
            df = pd.DataFrame(records, columns=['Code',
                'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 
                'Turnover', 'Amplitude', 'Change_Pct', 'Change_Amt', 'Turnover_Rate'
            ])
            
            for col in df.columns[2:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"数据解析失败: {e}")
            return pd.DataFrame()
        
    def get_stock_code_list(page_size=50, fn='f9', po = 0):
        """
        请求东方财富的股票列表接口，获取指定数量的涨幅榜股票。

        Field ID (fid),含义
        f2,最新价 / 现价 (Latest Price),价格最高的股票在前。
        f3,涨跌幅 (Change Pct),涨幅最大的股票在前 (涨幅榜)。
        f5,成交量 (Volume),成交量最大的股票在前。
        f6,成交额 (Turnover),成交额最大的股票在前。
        f8,换手率 (Turnover Rate),换手率最高的股票在前。
        f9,市盈率 (PE Ratio),市盈率最低的股票在前。
        f10,市净率 (PB Ratio),市净率最低的股票在前

        :param page_size: 每页返回的股票数量
        :param fn: 排序依据字段ID
        :param po: 排序方式 （1 默认 0 相反）
        """
        
        params = {
            'np': 1,
            'fltt': 1,
            'invt': 2,
            'cb': f'jsonp_list_{int(time.time() * 1000)}', 
            'fs': 'm:1+t:23+f:!2', 
            'fields': 'f12,f14,f3,f2,f152,f5,f6,f7,f15,f18,f16,f17,f10,f8,f9,f23',
            'fid': fn,
            'pn': 1,
            'pz': page_size, # 设置页大小
            'po': po,        # 排序方式
            'dect': 1,
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'wbp2u': f'{random.randint(10**15, 10**16)}|0|1|0|web', # 随机化wbp2u
            '_': int(time.time() * 1000)
        }
        
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        headers = generate_random_headers("https://quote.eastmoney.com/center/gridlist.html")

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            # 处理 JSONP 格式数据
            jsonp_text = response.text
            start_index = jsonp_text.find('{')
            end_index = jsonp_text.rfind('}') + 1
            json_data = jsonp_text[start_index:end_index]
            data = json.loads(json_data)
            
            # 提取股票代码列表
            stock_codes = []
            if 'data' in data and data['data'] and 'diff' in data['data']:
                for item in data['data']['diff']:
                    code = str(item.get('f12', '')).zfill(6) # f12 是股票代码
                    if code:
                        stock_codes.append(code)
                return stock_codes
                
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
        except Exception as e:
            print(f"数据解析失败: {e}")
            
        return []


    def get_stock_data(data_path, stock_codes, length = 1000):
        all_data = []
        for code in stock_codes:
            print(f"get {code} data...   ", end="")
            df = spiderTool.get_stock_kline_data(code, '1', length)
            if not df.empty:
                print(f"successfully got {len(df)} records.")
                all_data.append(df)
            else:
                print(f"some thing wrong when getting {code} data.")
            time.sleep(random.uniform(1, 3)) 
        result = pd.concat(all_data, ignore_index=True)
        result.to_csv(data_path, index=False, encoding='utf-8-sig')
        # result.to_parquet(data_path, index=False)
        print("*** All data saved. total records: ", len(result), " ***")


# test
if __name__ == '__main__':
    ls = spiderTool.get_stock_code_list(page_size=10)
    print("获取到的股票代码列表:", ls)
    df = spiderTool.get_stock_kline_data(ls[0], '1', limit=10)
    if not df.empty:
        print("数据前20行:")
        print(df.head(20))
        print(f"总共获取了 {len(df)} 条数据。")