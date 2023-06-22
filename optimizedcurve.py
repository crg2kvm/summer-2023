import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
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
    if start is None:
        data = yf.download(stocks,period=timeframe)
    else:
        data = yf.download(stocks, start=start, end=end)
    close_prices = data['Close']
    price_relative = close_prices / close_prices.shift(1)
    log_returns = np.log(price_relative).dropna() * 100
    # Convert to annual returns
    expected_returns = ((1 + log_returns.mean()/100)**252 - 1) * 100
    cov_matrix = log_returns.cov() 
    return cov_matrix, expected_returns.values

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
    cov_matrix, stockreturns = allto(stocks, timeframe,start,end)
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

    #print("Success: ", result.success)
    #print("Message: ", result.message)
    #plt.scatter(efficient_portfolio_volatilities, efficient_portfolio_returns)
    #plt.xlabel('Volatility')
    #plt.ylabel('Expected Returns')
    #plt.title('Efficient Frontier')
    #plt.show()
    if stock_weight is None:
        return efficient_portfolio_returns,efficient_portfolio_volatilities, efficient_portfolio_weights
    else:
        return efficient_portfolio_returns,efficient_portfolio_volatilities


"""
This function uses the efficient_frontier() function to construct efficient frontiers for
 a portfolio under different constraints. It then plots these frontiers on a graph.
"""

def graphit(portfolios,stocks, security_type, time_frame, noconstraints = False, start = None, end = None):
    start1 = time.perf_counter()
    if noconstraints == False:
        weights = np.random.random(len(stocks))
        weights = [1/len(stocks)] * len(stocks)
        z = [1/6,1/6,1/6,1/6,1/6,1/6]
        x2, y2 = efficient_frontier(stocks, portfolios, time_frame,z,security_type,0.60,0.40)
        x1, y1 = efficient_frontier(stocks, portfolios, time_frame,z,security_type,0.55,0.45)
        test = pd.DataFrame()
        one = np.ones(portfolios)
        zero = np.zeros(portfolios)
        #test["returns"] = np.append(x1,x2)
        #test["risk"] = np.append(y1,y2)
        #test["colors"] = np.append(zero,one)
        plt.plot(y2,x2,label = "60-40")
        plt.plot(y1,x1,label = "55-45")
        plt.legend()
        plt.show()
    else:
        weights = np.random.random(len(stocks))
        weights = [1/len(stocks)] * len(stocks)
        z = weights
        
        x2, y2 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.60,0.40, start, end)
        x1, y1 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.55,0.45, start, end)
        x5, y5 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.7,0.3, start, end)
        x4, y4 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.5,0.5, start, end)
        x6, y6 = efficient_frontier(stocks, portfolios, time_frame,security_type,0.4,0.6, start, end)
        x3, y3,w1 = efficient_frontier(stocks, portfolios * 10, time_frame,start= start, end=end)
        bondcount = security_type.count("Bond")
        colors = []
        for i in w1:
            colors.append(np.sum(i[:bondcount]))
        
        plt.plot(y2,x2,label = "60-40")
        plt.plot(y1,x1,label = "55-45")
        plt.plot(y5,x5,label = "70-30")
        plt.plot(y4,x4,label = "50-50")
        plt.plot(y6,x6,label = "40-60")
        end1 = time.perf_counter()
        print(end1-start1)

        plt.scatter(y3,x3,label = "No Constraints",c=colors,marker=".",cmap="RdYlGn",s=5)
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
graphit(100,assets,assettype,"10y",True,"2016-01-01","2020-01-01")

#graphit(1000,["TLT","AGG","SHY","XLP","XLE","XOP"],["Bond","Bond","Bond","Stock","Stock","Stock"],"ytd",True
