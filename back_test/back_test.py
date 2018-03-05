# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:21:37 2018

@author: Administrator
"""

# 选定使用特斯拉两年的股票走势数据
klpd = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)
# 头一年（[:252]）作为训练数据, 美股交易中一年的交易日有252天
trainkl = klpd[:252]
# 后一年（[252:]）作为回测数据
testkl = klpd[252:]

# 分别画出两部分数据收盘价格曲线
tmpdf = pd.DataFrame(np.array([trainkl.close.values,testkl.close.values]).T,
                    columns = ['train','test'])
tmpdf[['train','test']].plot(subplots=True,grid=True,
                            figsize=(14,7))

                            
# 训练数据的收盘价格均值
closemean=trainkl.close.mean()
# 训练数据的收盘价格标准差
closestd = trainkl.close.std()

# 构造卖出信号阀值
sellsignal = closemean + closestd/3
# 构造买入信号阀值
buysignal = closemean - closestd/3

# 可视化训练数据的卖出信号阀值，买入信号阀值及均值线
plt.figure(figsize=(14,7))
# 训练集收盘价格可视化
trainkl.close.plot()
# 水平线，买入信号线, lw代表线的粗度
plt.axhline(buysignal,color='r',lw=3)
# 水平线，j均值线
plt.axhline(closemean,color='black',lw=1)
# 水平线，卖出信号线, lw代表线的粗度
plt.axhline(sellsignal,color='r',lw=3)
plt.legend(['train close', 'buy_signal', 'close_mean', 'sell_signal'],
          loc='best')
plt.show()

# 将卖出信号阀值，买入信号阀值代入回归测试数据可视化
plt.figure(figsize=(14,7))
# 测试集收盘价格可视化
testkl.close.plot()
# buysignal直接代入买入信号
plt.axhline(buysignal,color='r',lw=3)
# 水平线，j均值线
plt.axhline(closemean,color='black',lw=1)
# sell_signal直接代入卖出信号
plt.axhline(sellsignal,color='r',lw=3)
plt.legend(['test close', 'buy_signal', 'close_mean', 'sell_signal'],
           loc='best')
plt.show()

print('买入信号阀值:{} 卖出信号阀值:{}'.format(buysignal,sellsignal))                        




#回测函数***************************************************************************************
def back_test(hist_data,signal):
    testkl = hist_data
    # 寻找测试数据中满足买入条件的时间序列
    buy_Index = hist_data[signal == 1].index
    # 将找到的买入时间系列的信号设置为1，代表买入操作
    hist_data.loc[buyIndex,'signal'] = 1
    #表7-2所示
    testkl[52:57]
    
    
    # 寻找测试数据中满足卖出条件的时间序列
    sellIndex = testkl[testkl['close'] >= sellsignal].index
    
    # 将找到的卖出时间系列的信号设置为0，代表卖出操作
    testkl.loc[sellIndex,'signal'] = 0
    # 表7-3所示
    testkl[48:53]
    
    # 由于假设都是全仓操作所以signal＝keep，即1代表买入持有，0代表卖出空仓
    testkl['keep'] = testkl['signal']
    # 将keep列中的nan使用向下填充的方式填充，结果使keep可以代表最终的交易持股状态
    testkl['keep'].fillna(method='ffill',inplace=True)
    testkl[50:56]
    
    # 由于假设都是全仓操作所以signal＝keep，即1代表买入持有，0代表卖出空仓
    testkl['keep'] = testkl['signal']
    # 将keep列中的nan使用向下填充的方式填充，结果使keep可以代表最终的交易持股状态
    testkl['keep'].fillna(method='ffill',inplace=True)
    testkl[50:56]
    
    #相当于创建了一个滤波器，过滤掉输入信号为0的结果
    testkl['trendProfit'] = testkl['keep']*testkl['benchmarkProfit']
    #最后计算两者的累计收益并可视化对比
    testkl[['benchmarkProfit','trendProfit']].cumsum().plot(grid=True,figsize=(14,7))
    
    
    #度量的基本使用方法
    from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop
    from abupy import abu, ABuProgress, GridSearch, ABuFileUtil, ABuGridHelper, AbuMetricsBase, AbuBlockProgress
        
    # 设置初始资金数
    read_cash = 1000000
    # 设置选股因子，None为不使用选股因子
    stock_pickers = None
    # 买入因子依然延用向上突破因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]
    
    # 卖出因子继续使用上一章使用的因子
    sell_factors = [
        {'stop_loss_n': 1.0, 'stop_win_n': 3.0,
         'class': AbuFactorAtrNStop},
        {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
        {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    ]
    # 择时股票池
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    # 使用run_loop_back运行策略
    abu_result_tuple, kl_pd_manager = abu.run_loop_back(read_cash,
                                                       buy_factors,
                                                       sell_factors,
                                                       stock_pickers,
                                                       choice_symbols=
                                                       choice_symbols,
                                                       n_folds=2)
    ABuProgress.clear_output()
    
    #夏普比率
    def sharpe(rets, ann=252):
        return rets.mean() / rets.std() * np.sqrt(ann)
    
    print('策略sharpe值计算为＝{}'.format(sharpe(metrics.algorithm_returns)))