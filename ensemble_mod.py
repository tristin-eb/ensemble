'''Code below contains everything to run weekly model'''

# Install packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pymc3 as pm
import theano
import itertools

def fourier_series(n, period, series_order):
    '''
    df used to get length is the length of the series I want to match;
    Period is how long the seasonal periods last (i.e. 365.25 days for year)
    Series order is how many terms I want to fit (matrix will be 2*series_order)
        Increasing order allows series to change more quickly; this should be tuned
    This gets called in the following function (not called by user)
    '''
    # n = df.shape[0]
    t = np.arange(1,n+1)
    return pd.DataFrame(np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)]))

def add_fourier(df,end_date,yr_order,week_order):
    '''
    Add fourier series to data frame
    yearly and weekly are bools; specify yr and week order
    Need a time/date variable for the index
    periods refers to time in future user wants to return
    '''
    end = pd.to_datetime(end_date)
    idx = pd.date_range(start=df.index[0], end = end_date, freq='D')
    n = len(idx)

    if yr_order>0 and week_order>0:
        yr_four = fourier_series(n,365.25,yr_order) #year set to 365.25 days
        week_four = fourier_series(n,7,week_order) # week set to 7 days
        four_dat = pd.DataFrame(pd.concat([yr_four,week_four],axis=1)) # concat year, week
        four_dat.index = idx#pd.to_datetime(df.index) # First column is the old index of df, now reset

    elif yr_order>0 and week_order==0:
        yr_four = fourier_series(n,365.25,yr_order)
        four_dat= pd.DataFrame(yr_four)
        four_dat.index = idx#pd.to_datetime(df.index)

    elif yr_order==0 and week_order>0:
        week_four = fourier_series(n,7,week_order)
        four_dat= pd.DataFrame(week_four)
        four_dat.index = idx#pd.to_datetime(df.index)

    else:
        return ['No seasonal var requested']
    return(four_dat)


def holiday_to_df(df,end_date,country_hol= ''):
    '''
    using package from datetime;
    local var 'call' is to package (i.e. 'holiday_to_df(holidays.AR(years=yr)')
    Make sure country is in dat
    Time var needs to be time/date index!
    '''

    import holidays # for holidays
    end = pd.to_datetime(end_date)
    idx = pd.date_range(start=df.index[0], end = end_date, freq='D')
    n = len(idx)

    yr = [i for i in idx.year.unique()] # get years from timevar in df
    if country_hol.upper() == 'AR':
        call = holidays.AR(years=yr)
    elif country_hol.upper() == 'AU':
        call = holidays.AU(years=yr)
    elif country_hol.upper() == 'BE':
        call = holidays.BE(years=yr)
    elif country_hol.upper() == 'CA':
        call = holidays.CA(years=yr)
    elif country_hol.upper() == 'DE':
        call = holidays.DE(years=yr)
    elif country_hol.upper() == 'ES':
        call = holidays.ES(years=yr)
    elif country_hol.upper() == 'GB':
        call = holidays.UK(years=yr)
    elif country_hol.upper() == 'IE':
        call = holidays.IE(years=yr)
    elif country_hol.upper() == 'NL':
        call = holidays.NL(years=yr)
    elif country_hol.upper() == 'NZ':
        call = holidays.NZ(years=yr)
    elif country_hol.upper() == 'ROE':
        call = holidays.ECB(years=yr)
    elif country_hol.upper() == 'US':
        call = holidays.US(years=yr)
    else:
        return ['No holidays for ' + country_hol ]
    dates = []
    names = []
    try: # because not all countries have holidays this may be buggy...
        for date, name in sorted(call.items()):
            dates.append(pd.to_datetime(date))
            names.append(name)
        hols = pd.DataFrame({'date':dates, 'holiday':names,}) # may not need to return this; just need country for date
        hols = hols[hols['date']<=end] # don't return holidays outside of the future period
        hols.index = hols['date']
        hols = hols.drop(columns='date')
        return(hols)
    except:
        pass


def get_X(df,end_date,country_hol,hol_dummies,yr_order, week_order):
    '''
    Create df used for X vars
    Requires two functions above: 1) add_fourier, 2) holiday_to_df
    Hack: if you don't want the holidays, just put in a country th
    Holl dummies is true or false

    '''
    end_date = pd.to_datetime(end_date)
    four_df = add_fourier(df,end_date,yr_order,week_order)
    if hol_dummies==True:
        hol_df = pd.get_dummies(holiday_to_df(df,end_date,country_hol))
    elif hol_dummies == False:
        hol_df = holiday_to_df(df,end_date,country_hol)
    else:
        return(['hol_dummies must be True or False','hol_dummies must be True or False'])
        pass
    if len(hol_df) > 1 and len(four_df)>1:
        merged_df = four_df.merge(hol_df,how='left',left_index=True,right_index=True)
        merged_df = merged_df.fillna(0)
#         return merged_df
    elif len(four_df)>1:
#         return four_df
        merged_df=four_df
    elif len(hol_df)>1:
        merged_df = hol_df
    else:
        return(['No seasons or Holidays for ' + country_hol, 'No seasons or Holidays for ' + country_hol])
        pass
    X_train = merged_df[merged_df.index <= df.index[-1]] # checked that this ends on the correct day
    X_test = merged_df[merged_df.index > df.index[-1]] # This begins one day after X_train

    if X_test.shape[0] ==0 and hol_dummies == True: # if no holidays, returns a matrix with no vars
        test_idx = pd.date_range(start = df.index[-1] + pd.Timedelta('1 day'), end =end_date, freq='D')
        X_test = pd.DataFrame(0, index=test_idx, columns=X_train.columns)
    return([X_train, X_test])


# Consecutive days
def consec_days(df):
    '''
    Remove non-consecutive days from dataframe (these are early days, much has changed since)
    Just do this with DV, then remove outliers on DV. Then merge with X_train
    Be careful with this when there is no Fourier series (no consecutive holidays!!!)
    Maybe don't let the function work if there is no Fourier?
    '''
    idx = df.index
    last_day = pd.to_datetime('2013-01-01') # This is first day in query, so should be after this day if non-consec
    for i in range(len(idx)):
        if idx[i] - idx[i-1]  > pd.Timedelta('1 day') and idx[-1] - idx[i] > pd.Timedelta('365 days'):
            last_day=idx[i] # update last day
    new_df = df[df.index>=last_day] # new dataframe all consecutive
    return(new_df)


def remove_outliers(df,sds=3):
    '''
    Outlier defined as standard deviations from the mean
    User can specify how many standard devs constitute an outlier (default is 3)
    Only do this when there is a variable to call (may not be the same across columns, but probably is...)
    '''
    new_dat = df[(df - df.mean()).abs() < sds*df.std()]
    return(pd.DataFrame(new_dat))

def y_x_merge(y,X):
    '''
    Take the cleaned/consecutive day removed y and make sure that X matches this.
    Return an X and a y clean

    Now clean all data first (don't let this be optional)
    '''
    y = consec_days(y)
    y = remove_outliers(y,sds=3)
    merged = y.merge(X, how='inner', left_index=True, right_index=True)
    y_clean = pd.DataFrame(merged.iloc[:,0])
    X_clean = pd.DataFrame(merged.iloc[:,1:])
    return([y_clean,X_clean])


# Now remove seasonality/holidays for mods that don't take exogenous vars
import statsmodels.api as sm
def adj_params(X,y):
    '''
    Find effect of season/holidays on trend
    This just finds the parameters,
    '''
    X = sm.add_constant(X) # This is only done locally; could do this to X var
    try:
        mod = sm.OLS(y,X).fit() # change OLS to GLM to get MLE estimate
        params = mod.params # will need to multiply params by new X vals later
    except:
        new_x = pd.get_dummies(X['holiday'],drop_first=True)
        X = X.drop(columns='holiday')
        X = pd.concat([X,new_x],axis=1)
        mod = sm.OLS(y,X).fit()
        params = mod.params # will need to multiply params by new X vals later

    return(params)

def adj_vars(X, params):
    '''
    Use params from previous function to account for seasonal and holiday effects
    Will need this twice, once for training and one for test series
    '''
    try:
        adj_vars = np.dot(sm.add_constant(X),params)
    except:
        new_x = pd.get_dummies(X['holiday'],drop_first=True)
        X = X.drop(columns='holiday')
        X = pd.concat([X,new_x],axis=1)
        adj_vars = pd.DataFrame(np.dot(sm.add_constant(X),params))
        adj_vars.index = X.index
    return(adj_vars)

### Now run models

from statsmodels.tsa.api import Holt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics


def est_models(df,end_date,country_hol,hol_dummies,yr_order,week_order):
    '''
    Put above together and run a bunch of models
    '''

    # Set up vars for models that take exog vars for seasons/holidays
    X_train, X_test = get_X(df, end_date, country_hol, hol_dummies, yr_order, week_order)
    dv, X_train = y_x_merge(df,X_train)

    # ARIMA
    print('Estimating mod 1 of 7')
    try: # MA term sometimes leads to a non-invertible matrix
        arima_mod = sm.tsa.SARIMAX(dv,exog=X_train, order=(2,1,1),seasonal_order=(0,0,0,0)).fit()
    except:
        arima_mod = sm.tsa.SARIMAX(dv,exog=X_train, order=(2,1,0),seasonal_order=(0,0,0,0)).fit()
    arima_fcst = arima_mod.forecast(X_test.shape[0], exog=X_test) # periods to forecast is length of X_test
    arima_fcst.index = X_test.index

    arima_fitted = arima_mod.predict()
    arima_fitted.index = X_train.index


#      OLS
    print('Estimating mod 2 of 7')
    ols_trend = pd.DataFrame({'t':[i for i in range((X_train.shape[0]+X_test.shape[0]))]}) # create trend var through pred_range
    ols_df = sm.add_constant(ols_trend) # add constant
    ols_df_cut = ols_df.iloc[:-X_test.shape[0],:] # remove pred range to estimate within sample
    ols_df_cut.index = X_train.index
    ols_df_cut = pd.concat([ols_df_cut, X_train], axis=1) # add exogenous IVs from X_train

    ols_df_cut.index = dv.index # make X index match y index
    ols_mod = sm.OLS(dv,ols_df_cut).fit() # fit model
    ols_df_fut = ols_df.iloc[-X_test.shape[0]:,:]
    ols_df_fut.index = X_test.index
    ols_df_fut = pd.concat([ols_df_fut,X_test],axis=1)

    ols_forecast = ols_mod.predict(ols_df_fut) #forecast based on future range
    ols_forecast.index= pd.to_datetime(arima_fcst.index)

    ols_fitted = ols_mod.fittedvalues
    ols_fitted.index = X_train.index

    # OLS with cubic polynomial
    print('Estimating mod 3 of 7')
    ols_trend = pd.DataFrame({'t':[i for i in range((X_train.shape[0]+X_test.shape[0]))]}) # create trend var through pred_range
    ols_trend['t2'] = ols_trend['t']**2
    ols_trend['t3'] = ols_trend['t']**3

    ols_cubic = sm.add_constant(ols_trend)
    ols_cubic_cut = ols_cubic.iloc[:-X_test.shape[0],:]
    ols_cubic_cut.index = X_train.index
    ols_cubic_cut = pd.concat([ols_cubic_cut, X_train], axis=1)
    ols_cubic_cut.index = dv.index
    ols_cubic_mod = sm.OLS(dv,ols_cubic_cut).fit()

    ols_cubic_fut = ols_cubic.iloc[-X_test.shape[0]:,:]
    ols_cubic_fut.index = X_test.index
    ols_cubic_fut = pd.concat([ols_cubic_fut,X_test],axis=1)

    ols_cubic_forecast = ols_cubic_mod.predict(ols_cubic_fut) #forecast based on future range
    ols_cubic_forecast.index = arima_fcst.index

    ols_cubic_fitted = ols_cubic_mod.fittedvalues #forecast based on future range
    ols_cubic_fitted.index = X_train.index

    ## SARIMA -- needs different seasonal vars (no weekly)
    print('Estimating mod 4 of 7')
    X_train_s, X_test_s = get_X(df, end_date, country_hol, hol_dummies, yr_order, week_order =0)
    dv, X_train_s = y_x_merge(df,X_train_s)

    try:
        sarima_mod = sm.tsa.SARIMAX(dv,exog=X_train_s, order=(2,1,1),seasonal_order=(1,0,1,7)).fit()
    except:
        sarima_mod = sm.tsa.SARIMAX(dv,exog=X_train_s, order=(2,1,0),seasonal_order=(1,0,1,7)).fit()
    sarima_fcst = sarima_mod.forecast(X_test.shape[0],exog=X_test_s) #+ adj_fut_s
    sarima_fcst.index = arima_fcst.index

    sarima_fitted = sarima_mod.predict()
    sarima_fitted.index = X_train_s.index

    ## HW -- needs holiday adjustments to DV and yearly seasonality;
    ## Data adjusted rather than given as exogenous vars as in SARIMA (_s refers to SARIMA from above)
    print('Estimating mod 5 of 7')
    pars = adj_params(X_train_s,dv) # Get parameters to adjust
    adj_pars = adj_vars(X_train_s,pars)
    adj_pars_fut = adj_vars(X_test_s,pars)
    y_hw = dv.iloc[:,0] - adj_pars # adjusting dv for holidays; year seeasoanlity
    hw_mod = sm.tsa.ExponentialSmoothing(y_hw, trend='add',seasonal='add',
                                      seasonal_periods=7).fit() # want exponential smoothing
    hw_fcst = hw_mod.forecast(X_test_s.shape[0]) + adj_pars_fut
    hw_fitted = hw_mod.fittedvalues + adj_pars

    hw_fcst.index = arima_fcst.index

    # Prophet -- needs holidays, but no seasonality
    print('Estimating mod 6 of 7')
    X_train_p,X_test_p = get_X(df, end_date, country_hol, hol_dummies=False, yr_order=0, week_order=0)
    # just want the length for forecast (holidays aren't at daily level), use len from sarima
    fcst_len = X_test_s.shape[0]
    if type(X_train_p) is not str:
        dv_p, X_train_p = y_x_merge(df,X_train_p) # don't use prophet DV!
        proph_hol = pd.concat([X_train_p, X_test_p],axis=0)
        proph_hol['ds']  = proph_hol.index
        proph_hol = proph_hol[['ds','holiday']]
        proph_dat = pd.DataFrame({'ds':dv.index,'y': dv.iloc[:,0]})
        m = Prophet(seasonality_mode = 'multiplicative',holidays = proph_hol).fit(proph_dat) # holidays=holidays goes in first call
    else:
        proph_dat = pd.DataFrame({'ds':dv.index,'y': dv.iloc[:,0]})
        m = Prophet(seasonality_mode = 'multiplicative').fit(proph_dat) # holidays=holidays goes in first call

    future = m.make_future_dataframe(periods = fcst_len,freq='D')
    forecast = m.predict(future)
    prophet_fcst = forecast['yhat'][-fcst_len:]
    prophet_fcst.index = arima_fcst.index

    prophet_fitted = forecast['yhat'][:-fcst_len]
    prophet_fitted.index = X_train_s.index #prophet train only has holidays; not daily


    # structural time series with local linear; here just want holidays
    print('Estimating mod 7 of 7')

    sts_mod =  sm.tsa.UnobservedComponents(dv, 'local level', trend='local linear trend', cycle=True,
                                        damped_cycle=True,
                                        stochastic_cycle=True,exog=X_train_s, seasonal=7).fit()
    sts_fcst = sts_mod.forecast(X_test_s.shape[0], exog=X_test_s)
    sts_fcst.index = arima_fcst.index

    sts_fitted = sts_mod.predict()
    sts_fitted.index = X_train_s.index

    fitted = pd.DataFrame({'ARIMA':arima_fitted,'OLS':ols_fitted,'OLS_cubic':ols_cubic_fitted,
                       'SARIMA': sarima_fitted, 'HW':hw_fitted,'Prophet': prophet_fitted,'STS':sts_fitted}) #


    fcst = pd.DataFrame({'ARIMA':arima_fcst,'OLS':ols_forecast,'OLS_cubic':ols_cubic_forecast,
                       'SARIMA': sarima_fcst, 'HW':hw_fcst,'Prophet': prophet_fcst,'STS':sts_fcst}) #


    return([fitted, fcst, dv]) # Shape of DV changes; make sure it is correct when passing below


def first_cuts(df, var, end_date):
    '''Probably not the best name, but runs the whole model outputs estimates as well as samples'''

    # Not going through all geos for now, so choosing a few
    df['event_geo'] = df['event_geo'].replace(['AU', 'BE', 'DE', 'ES', 'GB', 'IE', 'NL', 'NZ', 'ROE','ROW', 'BR'],'INT')

    geos = df.event_geo.unique().tolist() # put those few in a list
    new_ret = ['new', 'returning']
    pred_vars = [var] # Only need paid items here

    cuts = [i for i in itertools.product(geos,new_ret,pred_vars)]
    # Now forecast over all cuts of geo, new.returning and var
    fit_out = pd.DataFrame()
    fcst_out = pd.DataFrame()
    dv_out = pd.DataFrame()
    for cut in cuts:
        print('estimating: ', cut[0], cut[1], cut[2])

        dat = df[(df['event_geo']==cut[0]) & (df['new_returning']==cut[1]) ]
        dat['trx_date'] = pd.to_datetime(dat['trx_date'])
        dat = dat.groupby('trx_date').sum()
        dat.index = pd.to_datetime(dat.index)

#     dat = dat[dat.index<'2018-08-01'] # can change this for backtesting

        # Run fitted from above
        # VARIABLE IS LOG +1 HERE!!!!!!!!!!!!!!!!
        fitted, preds, dv = est_models(np.log1p(dat[cut[2]]),end_date,country_hol=cut[0],
                                       hol_dummies=True,yr_order=10,week_order=4)
        fitted['geo'] = cut[0]
        fitted['new_ret'] = cut[1]
        fitted['var'] = cut[2]
        preds['geo'] = cut[0]
        preds['new_ret'] = cut[1]
        preds['var'] = cut[2]

        dv['geo'] = cut[0]
        dv['new_ret'] = cut[1]
        dv['var'] = cut[2]


        fcst_out = pd.concat([fcst_out,preds]) # Just want preds here
        fit_out = pd.concat([fit_out,fitted]) # Just want preds here
        dv_out = pd.concat([dv_out,dv])

    # Organize above output for Bayesian model averaging

    # Make sure no NA in dv
    dv_var = dv_out[~np.isnan(dv_out[var])]#.drop(columns='gtf_usd')
    dv_var = dv_var.reset_index()
    fit_out = fit_out.reset_index()
    dat = fit_out.merge(dv_var, left_on = ['index','geo','new_ret','var'], right_on=['index','geo','new_ret','var'], how='inner')

    dat_var = dat[dat['var']==var]

    # Get grps for hierarchical model: geo, new/returning
    grps = dat.geo.unique()
    n_grps = len(grps)
    grp_lookup = dict(zip(grps,range(n_grps)))

    new_ret_grp = dat.new_ret.unique()
    n_new_ret = len(new_ret_grp)
    nr_grp_lookup = dict(zip(new_ret_grp,range(n_new_ret)))

    grp = dat_var['grp_code'] = dat_var.geo.replace(grp_lookup).values # geo vals
    nr_grp = dat_var['nr_grp_code'] = dat_var.new_ret.replace(nr_grp_lookup).values # new_ret vals

    # Get fitted values
    X = dat_var[['ARIMA', 'HW', 'OLS', 'OLS_cubic', 'Prophet', 'SARIMA', 'STS']]

    ### standardize variables to run faster
    X['ARIMA'] =( X['ARIMA'] - X['ARIMA'].mean())/X['ARIMA'].std()
    X['SARIMA'] =( X['SARIMA'] - X['SARIMA'].mean())/X['SARIMA'].std()
    X['HW'] = (X['HW'] - X['HW'].mean())/X['HW'].std()
    X['OLS'] =( X['OLS'] - X['OLS'].mean())/X['OLS'].std()
    X['OLS_cubic'] = (X['OLS_cubic'] - X['OLS_cubic'].mean())/X['OLS_cubic'].std()
    X['Prophet'] = (X['Prophet'] - X['Prophet'].mean())/X['Prophet'].std()
    X['STS'] = (X['STS'] - X['STS'].mean())/X['STS'].std()

    y =( dat[var] - dat[var].mean())/dat[var].std()

    # Now set vars, grp as shared vars
    arima = theano.shared(X.ARIMA.values) # These vals come from fitted above
    sarima = theano.shared(X.SARIMA.values)
    HW = theano.shared(X.HW.values)
    OLS = theano.shared(X.OLS.values)
    OLS3 = theano.shared(X.OLS_cubic.values)
    prophet = theano.shared(X.Prophet.values)
    STS = theano.shared(X.STS.values)

    grp2 = theano.shared(grp) # Will need to change these later as well
    nr_grp2 = theano.shared(nr_grp)

    y = y.values

    # Now run Bayesian model
    with pm.Model() as varying_intercept_slope:
        # Priors
        mu_a = pm.Normal('mu_a', mu=0., tau=0.0001)
        sigma_a = pm.HalfCauchy('sigma_a', 5)

        mu_b = pm.Uniform('mu_b', lower=0, upper=1.25) # Uniform prior on slope; max 1.25
        sigma_b = pm.HalfCauchy('sigma_b', 5) # prior on se of slope


        # random int/slope
        a = pm.Normal('a',mu=mu_a, sd=sigma_a, shape = (1, n_grps, n_new_ret))
        b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=(7, n_grps, n_new_ret))

        # Model error
        sd_y = pm.HalfCauchy('sd_y', 5)

        # Include multiple groups
        y_hat = a[0,grp2,nr_grp2] + b[0,grp2,nr_grp2] * arima + b[1,grp2,nr_grp2]*sarima + b[2,grp2,nr_grp2]*HW + \
        b[3,grp2,nr_grp2]*OLS + b[4,grp2,nr_grp2]*OLS3 + b[5,grp2,nr_grp2]*prophet + b[6,grp2,nr_grp2]*STS

        # Likelihood
        y_like = pm.Normal('y_like', mu=y_hat, sd=sd_y, observed=y)

        # Run trace
        varying_intercept_slope_trace = pm.sample(1000, tune=1000,chains=2) # Will want more tune and chains for automatic

    # Now run rest
    # 1) assign group-level vars to fcst

    fcst_var = fcst_out[fcst_out['var']==var]
    fcst_grp = fcst_var['grp_code'] = fcst_var.geo.replace(grp_lookup).values
    fcst_grp_nr = fcst_var['grp_code'] = fcst_var.new_ret.replace(nr_grp_lookup).values

    # Now reset shared variables, sample from PPC
    arima.set_value(fcst_var.ARIMA.values) # These come from forecast value
    sarima.set_value(fcst_var.SARIMA.values)
    HW.set_value(fcst_var.HW.values)
    OLS.set_value(fcst_var.OLS.values)
    OLS3.set_value(fcst_var.OLS_cubic.values)
    prophet.set_value(fcst_var.Prophet.values)
    STS.set_value(fcst_var.STS.values)

    grp2.set_value(fcst_grp) # make sure grp level covs fit
    nr_grp2.set_value(fcst_grp_nr)

    ppc = pm.sample_ppc(varying_intercept_slope_trace, model=varying_intercept_slope, samples=1000) # Just changed

    bayes_fcst = pd.DataFrame(ppc['y_like']).T
    bayes_mean = pd.DataFrame(bayes_fcst.median(1))
    bayes_mean['lwr'] = bayes_fcst.quantile(0.05,axis=1)
    bayes_mean['upr'] = bayes_fcst.quantile(.95,axis=1)
    bayes_mean.index = fcst_var.index

    bayes_mean.columns = ['est','lwr','upr']

    est_merge = pd.concat([fcst_var,bayes_mean],axis=1)

# est_merge

    est_merge[['est','lwr','upr']] = est_merge[['est','lwr','upr']].apply(np.exp) - 1 # For log1p
    est_merge[['ARIMA','HW','OLS','OLS_cubic','Prophet','SARIMA','STS']] = est_merge[['ARIMA','HW','OLS','OLS_cubic','Prophet','SARIMA','STS']].apply(np.exp)

    bayes_test  = np.exp(bayes_fcst) -1 # For log 1p

    return({'ests':est_merge,'samples':bayes_test})




# Now load data for training

eb_SSO = pd.read_csv('sso_dat.csv')
eb_SSO = eb_SSO[eb_SSO['migration_tag']=='None']

## Fit model

est_mod = first_cuts(eb_SSO, 'gtf_usd', '2021-12-31') # Go thru 2019

# Write mod to csv
est_mod.to_csv('filename.csv')
