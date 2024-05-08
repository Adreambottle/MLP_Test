import os
import sys
import bisect
import random
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from scipy import optimize
import mosek.fusion as mf

import tushare as ts
token = "xxxxxxxxxxxxxxxxxxxxxx"
ts.set_token(token)
pro = ts.pro_api()

def set_random_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def revalue_ignore_minimal(x, epsilon=1e-6):
    valid_ = x > epsilon
    x_v = x[valid_]
    x_v = x_v / x_v.sum()
    x_new = np.zeros(x.shape)
    x_new[valid_] = x_v
    return x_new

def get_input_value(df_pnl, nAlphas=-1, decay_ratio=None):
    if nAlphas == -1:
        nAlphas = df_pnl.shape[1]
    df_pnl = df_pnl.iloc[:, :nAlphas]
    r = df_pnl.mean().values
    Sigma = df_pnl.cov().values
    return r, Sigma

def sharpe_func(wts, r, Sigma, decay_ratio):
    if decay_ratio is not None:
        assert len(r) == len(decay_ratio)
        r = r * decay_ratio
    fracNumer = r @ wts
    fracDenom = np.sqrt(wts @ Sigma @ wts.T)
    sharpe = fracNumer / fracDenom
    return sharpe

class MaxSharpe():

    def __init__(self, r, Sigma, rf, decay_ratio=None, method="MOSEK", tolerance=1e-8):
        """
        r = np.mean(df_pnl, axis=0) # mean return of all alphas
        rf = 0. # risk free interest rate
        Sigma = np.cov(df_pnl, rowvar=False) # covariance of returns of all alphas
        """
        self.r = r
        self.rf = rf
        self.Sigma = Sigma
        self.tolerance = tolerance
        self.nAlpha = len(r)
        self.method = method.lower()

    def solve(self):
        if self.method == "mosek":
            print("Call MOSEK to solve the problem")
            r = self.r + 1.
            rf = self.rf + 1.
            Sigma = self.Sigma
            if not self.is_pos_def(Sigma):
                II = np.eye(self.nAlpha) * 1e-10
                Sigma = Sigma + II
            L_matrix = np.linalg.cholesky(Sigma)
            F = L_matrix.T

            try:
                wts = self.MaxSharpeMosek(r, rf, F, full=True, pos=True)
            except:
                print("MOSEK Fails, using SLSQP method")
                r = self.r
                rf = self.rf
                Sigma = self.Sigma
                tolerance = self.tolerance
                wts = self.MaxSharpeSLSQP(r, rf, Sigma, tolerance)

        if self.method == "slsqp":
            r = self.r
            rf = self.rf
            Sigma = self.Sigma
            tolerance = self.tolerance
            wts = self.MaxSharpeSLSQP(r, rf, Sigma, tolerance)

        return wts
    
    def MaxSharpeMosek(self, r, rf, F, full=True, pos=True):
        with mf.Model("Sharpe") as M:
            n = self.nAlpha
            y = M.variable("y", n. mf.Domain.greaterThan(0) if pos else mf.Domain.unbounded())
            z = M.variable("z", mf.Domain.greaterThan(0))

            t = M.variable()
            M.constraint(mf.Expr.vstack(t, mf.Expr.mul(F, y)), mf.Domain.inQCone())
            M.objective(mf.ObjectiveSense.Minimize, t)

            M.constraint(mf.Expr.sub(mf.Expr.dot(r, y), mf.Expr.mul(rf, z)), mf.Domain.equalsTo(1))

            if full:
                M.constraint(mf.Expr.sub(mf.Expr.sum(y), z), mf.Domain.equalsTo(0))

            M.solve()

            if M.getProblemStatus() == mf.ProblemStatus.PrimalAndDualFeasible and z.level()[0] > 0:
                zval = z.level()[0]
                return np.array([yi/zval for yi in y.level()])
            else:
                raise ValueError("No Solution or some issue")


    def MaxSharpeSLSQP(self, r, rf, Sigma, tolerance=1e-7):
        
        def neg_sharpe_func(x, r, Sigma, rf):
            fracNumer = r @ x.T - rf
            fracDenom = np.sqrt(x @ Sigma @ x.T)
            sharpe = fracNumer / fracDenom
            return -sharpe

        def constraintEq(x):
            A = np.ones(x.shape)
            b = 1
            constraintVal = np.matmul(A, x.T) - b
            return constraintVal
        
        nAlpha = self.nAlpha
        x_init = np.repeat(1/nAlpha, nAlpha)
        constraints = ({"type" : "eq", "fun":constraintEq})
        lb = 0.
        ub = 1.
        bounds = tuple([(lb, ub) for xi in x_init])

        opt = optimize.minimize(
            neg_sharpe_func,
            x0 = x_init,
            args = (r, Sigma, rf, nAlpha),
            method = "SLSQP",
            bounds = bounds,
            constraints = constraints,
            tol = tolerance
        )
        wts = opt.x
        return wts


def get_datelist(start: str, end: str, interval: int):
    df = pro.index_daily(ts_code='399300.SZ', start_date=start, end_date=end)
    date_list = list(df.iloc[::-1]['trade_date'])
    sample_list = []
    for i in range(len(date_list)):
        if i % interval == 0:
            sample_list.append(date_list[i])

    return sample_list


def main():


    # Generate pseudo data

    sta_date = "20210101"
    end_date = "20230101"
    pred_sta_date = "20220101"
    
    dates_str = np.array(get_datelist(sta_date, end_date, interval=1))
    pred_dates_str = dates_str[dates_str >= pred_sta_date]
    dates_dt = pd.to_datetime(dates_str)
    
    nDays = len(dates_dt)

    nAlpha = 20
    nStock = 3000
    alpha_mean = np.random.rand(nAlpha) / 20
    alpha_corr = np.random.rand(nAlpha, nAlpha) - 0.5
    for i in range(nAlpha):
        for j in range(nAlpha):
            if i < j:
                alpha_corr[i, j] = alpha_corr[j, i]
            elif i == j:
                alpha_corr[i, j] = 1.
    
    decay_ratio = 1 - np.random.rand(nAlpha) * 0.5

    alpha_cov = alpha_corr / 100.
    df_pnl = np.random.multivariate_normal(alpha_mean, alpha_cov, nDays)
    alpha_names = [f"alpha.{i}" for i in range(1, nAlpha+1)]
    

    df_pnl = pd.DataFrame(df_pnl, index=dates_str, columns=alpha_names)
    
    rf = 0.03 / 365
    rolling = False
    rolling_window = 180
    mean_decay = True
    cov_decay = False

    wts_list = []
    for tidx, today in tqdm(enumerate(pred_dates_str)):
        if rolling:
            train_index = dates_str[dates_str < today]
            train_len = len(train_index)
        else:
            train_index = dates_str[dates_str < today]
            train_len = len(train_index)
            
            train_index = train_index[train_len-rolling_window:train_len]
        
        df_pnl_sep = df_pnl.loc[train_index, :]
        decay_ratio = np.repeat(decay_ratio, train_len).reshape(nAlpha, -1).T
        df_pnl_sep_decay = df_pnl_sep * decay_ratio
        if mean_decay:
            r = df_pnl_sep_decay.means().values
        else:
            r = df_pnl_sep.means().values
        
        if cov_decay:
            Sigma = df_pnl_sep_decay.cov().values
        else:
            Sigma = df_pnl_sep.cov().values

        wts = MaxSharpe(r, Sigma, rf=rf, method="MOSEK").solve()
        wts_list.append(wts)

        fcst = np.random.randn(nStock, nAlpha)
        fcst = pd.DataFrame(fcst, columns=alpha_names)
        combo_fcst = fcst @ wts 
        combo_fcst.to_csv(f"{today}_combo_fcst.csv")
        

if __name__ == "__main__":
    main()