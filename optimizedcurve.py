import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from openpyxl import load_workbook
"""
This function calculates the expected returns
 and log returns for a given stock ticker over a specified time frame. 
 The time frame can be optionally bounded using start and end dates.
"""
def calcreturns(ticker,time_frame,start = None, end = None):
    tick = yf.Ticker(ticker)

    if start is None and end is None:
        #print("hit")
        dat = pd.DataFrame(tick.history(period=time_frame))
    else:
        dat = pd.DataFrame(tick.history(start=start, end = end))
    #print(dat)
    price_relative = []
    price_relative = np.array(np.zeros(len(dat)-2))
    for i in range(len(dat)-2):
        prior = dat['Close'][i]
        current = dat['Close'][i+1]
        price_relative[i] = current/prior
    returns1 = np.log(price_relative) * 100
    expected_returns = returns1.mean()
    return expected_returns, returns1


"""
This function computes the covariance matrix and returns for a 
list of stock tickers over a certain time frame. The time frame 
can be optionally bounded using start and end dates.
"""






def allto(stocks, timeframe, start=None, end=None):
    for stock in stocks:
        if start is None:
            data = yf.download(stocks,period=timeframe)
        else:
            data = yf.download(stocks, start=start, end=end)
        if data.empty:
                print(f"No data found for {stock} in the specified time frame. Skipping this asset...")
                continue
        close_prices = data['Close']
        price_relative = close_prices / close_prices.shift(1)
        log_returns = np.log(price_relative).dropna() * 100
        num_years = len(log_returns) / 252  # approximate number of trading days in a year
        """
        beginning_value = close_prices.iloc[0]
        ending_value = close_prices.iloc[-1]
        expected_returns = ((ending_value / beginning_value) ** (1 / num_years) - 1) * 100
        """
        expected_returns = ((1 + log_returns.mean()/100)**252 - 1) * 100
        cov_matrix = log_returns.cov() 
        return cov_matrix, expected_returns.values, num_years

"""
This function computes the variance (risk) of a portfolio given a set of portfolio weights and a covariance matrix.
"""

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))


"""
This function generates a set of weights for the assets in the portfolio. If specified, it can generate weights for a
 portfolio composed of stocks and bonds, with the portfolio weights for each group summing to certain specified amounts.
"""

def generate_portfolio_weights(num_assets, num_portfolios, num_bond=None, stock_weight=None, bond_weight=None):
    weights_matrix = []
    if num_bond is None:
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_matrix.append(weights)
    else:
        for i in range(num_portfolios):
            stock_weights = np.random.random(num_assets - num_bond)
            #print(stock_weights)
            stock_weights = stock_weight * (stock_weights/np.sum(stock_weights))
            bond_weights = np.random.random(num_bond)
            bond_weights = bond_weight * (bond_weights/np.sum(bond_weights))
            weights_matrix.append(np.concatenate((bond_weights,stock_weights)))
    return weights_matrix


"""
This function calculates the total return of a portfolio given a set of portfolio weights and the returns of the individual assets.
"""

def portfolio_return(weights, returns):
    return np.dot(weights, returns)


"""
This function constructs the efficient frontier for a portfolio of the given stocks. 
It calculates the returns and volatilities for different portfolio weights and returns these values. 
The function can also include constraints on the portfolio, such as a maximum weight for bonds and stocks.
"""

def efficient_frontier(stocks, num_portfolios, timeframe,  security_type = None, stock_weight = None, bond_weight = None, start = None, end = None):
    if security_type is not None: 
        security_type.sort()
        num_bond = security_type.count("Bond")
        weights_matrix = generate_portfolio_weights(len(stocks), num_portfolios, num_bond, stock_weight, bond_weight)
    else:
        weights_matrix = generate_portfolio_weights(len(stocks), num_portfolios)
    cov_matrix, stockreturns, num_years = allto(stocks, timeframe,start,end)
    efficient_portfolio_returns = []
    efficient_portfolio_volatilities = []
    efficient_portfolio_weights = []
    
    if security_type is not None:
        cons = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, stockreturns) - target},   
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x[:num_bond]) - bond_weight},  
                #{'type': 'eq', 'fun': lambda x: np.sum(x[num_bond:]) - stock_weight}
                )
        min_return = min(stockreturns[:num_bond]) * bond_weight + min(stockreturns[num_bond:]) * stock_weight
        max_return = max(stockreturns[:num_bond]) * bond_weight + max(stockreturns[num_bond:]) * stock_weight
        target_returns = np.linspace(min_return,max_return,num_portfolios)
    else:
        cons = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, stockreturns) - target},   
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                #{'type': 'ineq', 'fun': lambda x: portfolio_return(x, returns)}
                )
        target_returns = np.linspace(min(stockreturns), max(stockreturns), num_portfolios)
    for target in target_returns:

        bounds = tuple((0, 1) for asset in range(len(stocks)))
        result = minimize(portfolio_variance, weights_matrix[0], args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100000, 'ftol': 1e-9})
        
        efficient_portfolio_returns.append(target)
        efficient_portfolio_volatilities.append(np.sqrt(result['fun'] * 252))
        efficient_portfolio_weights.append(result['x'])

    if stock_weight is None:
        return efficient_portfolio_volatilities, efficient_portfolio_returns, efficient_portfolio_weights
    else:
        return efficient_portfolio_volatilities, efficient_portfolio_returns, efficient_portfolio_weights


"""
This function uses the efficient_frontier() function to construct efficient frontiers for
 a portfolio under different constraints. It then plots these frontiers on a graph.
"""

def graphit(portfolios,stocks, security_type, time_frame, noconstraints = False, start = None, end = None, rolling = False, port_bond=None, port_stock=None):
    start1 = time.perf_counter()
    if rolling == True:
        weights = np.random.random(len(stocks))
        weights = [1/len(stocks)] * len(stocks)
        x1, y1, w1 = efficient_frontier(stocks, portfolios, time_frame, security_type,port_stock,port_bond,start,end)
        plt.plot(x1,y1,label = start)
        title = str(port_stock)+"-"+str(port_bond)+" over " +str(time_frame)
        plt.title(title)

        plt.legend()
    else:
        writer = pd.ExcelWriter('portfolio_weights.xlsx', engine='openpyxl')
        weights = np.random.random(len(stocks))
        weights = [1/len(stocks)] * len(stocks)
        z = weights
        x0, y0, w0 = efficient_frontier(stocks, portfolios, time_frame,security_type,0,1, start, end)
        x1, y1, w1 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.55,0.45, start, end)
        x2, y2, w2 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.60,0.40, start, end)
        x3, y3, w3 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.4,0.6, start, end)
        x4, y4, w4 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.5,0.5, start, end)
        x5, y5, w5 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.7,0.3, start, end)
        x6, y6,w6 = efficient_frontier(stocks, portfolios * 10, time_frame,start= start, end=end)
        x7, y7, w7 = efficient_frontier(stocks, portfolios, time_frame,security_type,1,0, start, end)
        df1 = pd.DataFrame(w1, columns=stocks)
        df1['Returns'] = y1
        df1['Risk'] = x1
        df2 = pd.DataFrame(w2, columns=stocks)
        df2['Returns'] = y2
        df2['Risk'] = x2
        df3 = pd.DataFrame(w3, columns=stocks)
        df3['Returns'] = y3
        df3['Risk'] = x3
        df4 = pd.DataFrame(w4, columns=stocks)
        df4['Returns'] = y4
        df4['Risk'] = x4
        df5 = pd.DataFrame(w5, columns=stocks)
        df5['Returns'] = y5
        df5['Risk'] = x5
        df6 = pd.DataFrame(w6, columns=stocks)
        df6['Returns'] = y6
        df6['Risk'] = x6
        df0 = pd.DataFrame(w0, columns=stocks)
        df0['Returns'] = y0
        df0['Risk'] = x0
        df7 = pd.DataFrame(w7, columns=stocks)
        df7['Returns'] = y7
        df7['Risk'] = x7


        # Write each dataframe to a different worksheet
        df0.to_excel(writer, sheet_name='0-100')
        df1.to_excel(writer, sheet_name='55-45')
        df2.to_excel(writer, sheet_name='60-40')
        df3.to_excel(writer, sheet_name='40-60')
        df4.to_excel(writer, sheet_name='50-50')
        df5.to_excel(writer, sheet_name='70-30')
        df6.to_excel(writer, sheet_name='No Constraints')
        df7.to_excel(writer, sheet_name='100-0')

        # Close the Pandas Excel writer and output the Excel file
        writer._save()

        bondcount = security_type.count("Bond")
        colors = []
        for i in w6:
            colors.append(np.sum(i[:bondcount]))

        #plt.plot(x0,y0,label = "0-100")
        plt.plot(x1,y1,label = "55-45")
        plt.plot(x2,y2,label = "60-40")
        plt.plot(x3,y3,label = "40-60")
        plt.plot(x4,y4,label = "50-50")
        plt.plot(x5,y5,label = "70-30")
        plt.plot(x7,y7,label = "100-0")
        end1 = time.perf_counter()
        print(end1-start1)
        plt.scatter(x6,y6,label = "No Constraints",c=colors,marker=".",cmap="RdYlGn",s=5)

        sp500 = yf.download("^GSPC", start=start, end=end)
        sp500_price_relative = sp500['Close'] / sp500['Close'].shift(1)
        sp500_log_returns = np.log(sp500_price_relative).dropna() * 100
        sp500_annual_returns = ((1 + sp500_log_returns.mean()/100)**252 - 1) * 100
        sp500_annual_volatility = sp500_log_returns.std() * np.sqrt(252)
        plt.scatter(sp500_annual_volatility, sp500_annual_returns, color='red', label='S&P 500')
        spy = yf.download("SPY", start=start, end=end)
        spy_price_relative = spy['Close'] / spy['Close'].shift(1)
        spy_log_returns = np.log(spy_price_relative).dropna() * 100
        spy_annual_returns = ((1 + spy_log_returns.mean()/100)**252 - 1) * 100
        spy_annual_volatility = spy_log_returns.std() * np.sqrt(252)
        plt.scatter(spy_annual_volatility, spy_annual_returns, color='blue', label='SPY')
        agg = yf.download("AGG", start=start, end=end)
        agg_price_relative = agg['Close'] / agg['Close'].shift(1)
        agg_log_returns = np.log(agg_price_relative).dropna() * 100
        agg_annual_returns = ((1 + agg_log_returns.mean()/100)**252 - 1) * 100
        agg_annual_volatility = agg_log_returns.std() * np.sqrt(252)
        plt.scatter(agg_annual_volatility, agg_annual_returns, color='green', label='AGG')

        if start is None:
            plt.title(time_frame)
        else:
            plt.title(start+" to "+end)
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.legend()
        plt.show()
        

assets = ["TLT","AGG","SHY","XLP","XLE","XOP","XLY","XLF","XLV","XLI","XLB","XLK","XLU"]
assettype = ["Bond","Bond","Bond","Stock","Stock","Stock","Stock","Stock","Stock","Stock","Stock","Stock","Stock"]
#graphit(100,assets,assettype,"10y",True,"2010-01-01","2020-12-31")

#graphit(1000,["TLT","AGG","SHY","XLP","XLE","XOP"],["Bond","Bond","Bond","Stock","Stock","Stock"],"ytd",True
def rolling(start, end, windowsize, bond,stock):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    window_size = windowsize  # years
    dates_range = pd.date_range(start_date, end_date, freq='YS') 
    print(dates_range)
    for current_year in dates_range:
        start = current_year - pd.DateOffset(years=window_size)
        end = current_year
        if start < start_date:
            continue
        start = start.strftime('%Y-%m-%d')
        end = end.strftime('%Y-%m-%d')
        #print(assettype)
        time_frame = str(window_size) + "y"
        graphit(100, assets, assettype, time_frame, True, start, end,True,bond,stock)
    plt.xlabel("Risk(STD)")
    plt.ylabel("Return(Annualized log returns)")
    time_frame += ".pdf"
    plt.savefig(time_frame,format="pdf")
    plt.show()
rolling("2015-01-01","2023-01-01",5,0.4,0.6)