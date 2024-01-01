#!/usr/bin/env python3

from   itertools import product

import QuantLib as ql


def price(t = 1, s = 100, k = 100, z = 0.2, r = 0.05, q = 0.0, w = -1, e = True):
    anchor = ql.Date(18,4,2023)
    ql.Settings.instance().setEvaluationDate(anchor)
    day_count = ql.Actual360()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    spot_handle = ql.QuoteHandle( ql.SimpleQuote( s ) )
    flat_ts = ql.YieldTermStructureHandle( ql.FlatForward( anchor, r, day_count ))
    dividend_yield = ql.YieldTermStructureHandle( ql.FlatForward( anchor, q, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle( ql.BlackConstantVol( anchor, calendar, z, day_count ))
    bsm_process = ql.BlackScholesMertonProcess( spot_handle, dividend_yield, flat_ts, flat_vol_ts )

    payoff = ql.PlainVanillaPayoff( ql.Option.Put if w < 0 else ql.Option.Call, k )
    maturity = anchor + ql.Period(round(t * 360.), ql.Days)

    if e:
        engine = ql.QdFpAmericanEngine( bsm_process, ql.QdFpAmericanEngine.highPrecisionScheme() )
        asset = ql.VanillaOption( payoff, ql.AmericanExercise( anchor, maturity ))
    else:
        engine = ql.AnalyticEuropeanEngine( bsm_process )
        asset = ql.VanillaOption( payoff, ql.EuropeanExercise( maturity ))

    asset.setPricingEngine(engine)
    return asset.NPV()


def _diff(key, order=1, bump=0.01, **kwargs):
    x = kwargs[key]
    
    kwargs[key] = (1 + bump) * x
    v_up = price(**kwargs)

    kwargs[key] = (1 - bump) * x
    v_down = price(**kwargs)

    kwargs[key] = x

    if order == 1:
        return (v_up - v_down) / (2 * bump * x) + 0
    elif order == 2:
        return (v_up - 2 * price(**kwargs) + v_down) / (bump * x)**2 + 0
    return None

def delta(**kwargs):
    return _diff('s', order=1, **kwargs)
def gamma(**kwargs):
    return _diff('s', order=2, **kwargs)
def rho(**kwargs):
    return _diff('r', order=1, **kwargs) / 100
def theta(**kwargs):
    return _diff('t', order=1, **kwargs) / 360
def vega(**kwargs):
    return _diff('z', order=1, **kwargs) / 100


def generate(fo, portfolio):
    if portfolio == 'qdfp':
        t = [1./12, 0.25, 0.5, 0.75, 1.]
        s = [25, 50, 80, 90, 100, 110, 120, 150, 175, 200]
        k = [100]
        z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        r = [0.02, 0.04, 0.06, 0.08, 0.1]
        q = [0.0, 0.04, 0.08, 0.12]
        w = [-1]
        e = [True]
    else:
        e = [True, False]
        k = [100]
        q = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
        r = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        s = [25, 50, 80, 90, 100, 110, 120, 150, 175, 200]
        t = [1./12, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
        w = [-1, 1]
        z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    def to_string(x):
        x_ = round(x, 6) + 0
        return f'{x_:.6f}'.rstrip('0')

    fo.write('expiry,spot,strike,volatility,interest_rate,dividend_rate,parity,exercise,price\n')
    #fo.write('expiry,spot,strike,volatility,interest_rate,dividend_rate,parity,exercise,price,delta,gamma,vega,theta,rgo\n')
    for option in product(t, s, k, z, r, q, w, e):
        t, s, k, z, r, q, w, e = option
        price_ = to_string(price(*option))
        #delta_ = to_string(delta(**kwargs))
        #gamma_ = to_string(gamma(**kwargs))
        #rho_ = to_string(rho(**kwargs))
        #theta_ = to_string(theta(**kwargs))
        #vega_ = to_string(vega(**kwargs))

        line = f'{t},{s},{k},{z},{r},{q},{"c" if w > 0 else "p"},{"a" if e else "e"},{price_}\n'
        #line = f'{t},{s},{k},{z},{r},{q},{"c" if w > 0 else "p"},{"a" if e else "e"},{price_},{delta_},{gamma_},{vega_},{theta_},{rho_}\n'

        print(line, end='')
        fo.write(line)

if __name__ == '__main__':
    dst_path = 'portfolio.csv'
    with open(dst_path, 'w') as fo:
        generate(fo, 'qdfp')
