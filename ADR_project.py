import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from matplotlib import ticker


# Assignment: : Analyse the Intraday Price Drift of European Stocks, Condtioned on their
# ADR (American Depositary Receipt) Price Changes

# ADR is a way that non-US companies' stocks can be traded in the US (in dollars). It means that
# the non-US companies do not have to list on US stock exchanges (which comes at a cost)

# define variables
START_DATE='2021-10-01'
END_DATE='2023-08-01'
INTERVAL='1h'
MORNING_OPEN= "08:00:00"
MORNING_CLOSE= "12:00:00"

# Universe with all European ADRs traded on NASDAQ and NYSE and format as
# DR Name | Exchange | Country | Ticker | MarketCap

# ADR_universe=pd.read_csv(r"C:\Users\Argyr\OneDrive\Documents\WEBBTrader_Project\ADR_universe.csv")
# ADR_universe= ADR_universe[['DR Name', 'Exchange','Country','Ticker',"Company Market Cap (USD)"]]
# ADR_universe=ADR_universe.sort_values(by="Company Market Cap (USD)",ascending=False)
#
# #filter by exchange
#
# NASDAQ_ADR=ADR_universe[ADR_universe['Exchange']=='NASDAQ']
# NYSE_ADR=ADR_universe[(ADR_universe['Exchange']=='NYSE')&(ADR_universe['Country']=='United Kingdom')]
#
# NYSE_ADR_TICKER=NYSE_ADR['Ticker'].to_numpy().tolist()


def plot_data(data,start_date=START_DATE,end_date=END_DATE):

    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)

    plt.xlabel("Date")
    plt.ylabel("Intraday Return")
    plt.title("Intraday ADR Returns over 2 years")
    plt.legend()
    plt.show()

def get_intraday_returns(ticker,start_date=START_DATE,end_date=END_DATE):
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    intraday_returns = (data['Close'] - data['Open']) / data['Open']
    intraday_returns = pd.DataFrame(intraday_returns)

    return intraday_returns

def get_hourly_returns(ticker):
    data=yf.download(ticker, interval="1h", start=START_DATE, end=END_DATE)['Open']

    if len(ticker)>1:
        hold = {}
        for col in data.columns:
            temp=pd.DataFrame()
            temp[col]=data[col][(data.index.hour >= 8) & (data.index.hour <=  12)]
            temp["Date"]=temp[col].index.date

            #these are cumulative returns by construction
            temp[col]=temp.groupby("Date")[col].transform(lambda x: (x - x.iloc[0]) / x.iloc[0])
            hold[col]=temp

    else:
        hold = {}
        data=data[(data.index.hour >= 8) & (data.index.hour <=  12)]
        data=pd.DataFrame(data)
        data["Date"]=data.index.date
        data["Returns"]=data.groupby('Date')['Open'].transform(lambda x: (x - x.iloc[0]) / x.iloc[0])
        hold[ticker[0]]=data[['Returns']]

    return hold

def adjust_data(data,idx):

    #sometimes there is missing information, this function
    #removes rows that are missing from one of the two dataframes

    if data.shape[0]>idx.shape[0]:
        difference=data.index.difference(idx.index).tolist()
        for date in difference:
            data = data.drop(date)

    elif data.shape[0]<idx.shape[0]:
        difference = idx.index.difference(data.index).tolist()
        for date in difference:
            idx = idx.drop(date)

    data=data-idx.values
    data=pd.DataFrame(data)

    return data

def adjust_EUR_data(data,idx,key):

    #adjusting the EUR data seperately so that the time index
    #can be retained for further use

    temp_data=data[key]

    if temp_data.shape[0]>idx.shape[0]:
        difference=temp_data.index.difference(idx.index).tolist()
        for date in difference:
            temp_data = temp_data.drop(date)

    elif temp_data.shape[0]<idx.shape[0]:
        difference = idx.index.difference(temp_data.index).tolist()
        for date in difference:
            idx = idx.drop(date)

    temp_data=pd.DataFrame(temp_data)
    temp_data=temp_data-idx.values
    temp_data["Date"]=temp_data.index.date

    return temp_data

def adjust_hourly_data(data_d: dict,idx: dict):

    idx_key=list(idx)
    idx_data=idx[idx_key[0]]
    hold={}

    for key in list(data_d):

        data=data_d[key][key]
        data=pd.DataFrame(data)
        data["Date"] = data.index.date
        hold[key]=adjust_EUR_data(data,idx_data,key)

    return hold

############################################################################################################

#to define the abnormality of the data, I will consider how many standard deviations from the mean the data
#is since we expect returns to be normally distributed

def z_scores_df(data):
    data_mean=data.mean()
    data_std=np.sqrt(data.var())
    z_scores={}

    for column in data.columns:
        z_scores[column] = (data[column] - data_mean[column]) / data_std[column]

    z_scores = pd.DataFrame(z_scores)

    return z_scores

def z_scores_plot(data,num_sigma=2):

    z_scores=z_scores_df(data)
    plt.figure(figsize=(10, 6))

    for column in z_scores.columns:
        plt.plot(z_scores.index, z_scores[column], label=column)
        plt.xlabel("Date")
        plt.ylabel("Z_score")
        plt.title("Z_scores of the ADR_adjsuted intraday returns")
        plt.legend()
        plt.grid(True)
        plt.fill_between(z_scores.index,num_sigma,z_scores[column].max(),color='green', alpha=0.2)
        plt.fill_between(z_scores.index,num_sigma,-num_sigma,color='grey', alpha=0.2)
        plt.fill_between(z_scores.index,-num_sigma,z_scores[column].min(),color='red', alpha=0.2)
        plt.show()

    for column in z_scores.columns:
        plt.hist(z_scores[column],bins=500,label=column)
        plt.legend()
        plt.show()

def z_scores_statistics(data,num_sigma):
    z_scores =z_scores_df(data)
    data_mean = z_scores.mean()
    data_std = np.sqrt(z_scores.var())
    counts={}

    for column in z_scores.columns:
        positive=np.sum(z_scores[column]>data_mean[column]+num_sigma*data_std[column])
        neutral=np.sum((z_scores[column]<=data_mean[column]+num_sigma*data_std[column])&(z_scores[column]>=data_mean[column]-num_sigma*data_std[column]))
        negative = np.sum(z_scores[column] < data_mean[column]-num_sigma*data_std[column])
        counts[column]=[positive,neutral,negative]

    counts_df=pd.DataFrame(counts)
    counts_df.index=["Positive","Neutral","Negative"]

    return  counts_df

###########################################################################################################################

def event_class(data,num_sigma=2):

    z_scores=z_scores_df(data)
    pos,neg,neu={},{},{}

    for column in z_scores.columns:
        positive = pd.DataFrame()
        negative = pd.DataFrame()
        neutral = pd.DataFrame()

        positive[column]=z_scores[column][z_scores[column]>num_sigma]
        neutral[column]=z_scores[column][(z_scores[column]>=-num_sigma)&(z_scores[column]<=num_sigma)]
        negative[column]=z_scores[column][z_scores[column]<-num_sigma]

        pos[column]=positive
        neg[column]=negative
        neu[column]=neutral

    return pos,neg,neu

def condition_data(underlying_data: dict,abnormal_data: dict):

    columns=list(abnormal_data)
    underlying_columns=list(underlying_data)
    hold={}

    for col,ucol in zip(columns,underlying_columns):

        temp = pd.DataFrame()
        indices = abnormal_data[col].index.date
        uindices = underlying_data[ucol].index.date
        idx=[index for index in indices if index in uindices]
        mask = np.isin(uindices, idx)
        underlying_data[ucol]=pd.DataFrame(underlying_data[ucol][mask])
        temp[ucol]=underlying_data[ucol][ucol]
        hold[ucol]=temp

    return hold

# def plot_expected_cum(data: dict):
#
#     pos = condition_data(data, positive)
#     neg = condition_data(data, negative)
#     columns = list(data)
#
#     for type,title in zip([pos,neg],["positive","negative"]):
#         for col in columns:
#
#             temp=type[col]
#             plt.scatter(temp.index,temp.values,label=col)
#             plt.title(col+" expected cumulative returns from "+title+" ADR returns")
#             plt.xlabel("Time")
#             plt.ylabel("Cumulative Return")
#             plt.legend()
#             plt.show()

def plot_day(key,type: dict,data: dict,title: str,max_subplots=15):

    data_plot = condition_data(data, type)[key]
    data_plot["Date"]=data_plot.index.date
    unique_days=data_plot.index.to_series().dt.date.unique()

    for i in range(len(unique_days)//6+int(len(unique_days)%6 !=0)):

        fig, axs = plt.subplots(2, 3, figsize=(4.5 * 3, 2 * 4))
        axs = axs.flatten()
        temp=unique_days[:6]
        unique_days=unique_days[6:]

        for count,day in enumerate(temp):

            temp=pd.DataFrame()
            mask=np.isin(data_plot["Date"],day)
            temp[day]=pd.DataFrame(data_plot[mask][key])
            axs[count].set_xticks(temp.index,labels=["0","1","2","3","4"])
            axs[count].set_xlabel("Hours since market open")
            axs[count].plot(temp.index,temp.values)
            axs[count].set_title(day)

            if count+1 > 6:
                break
            else:
                continue

        fig.supylabel("Expected conditioned cumalative returns")
        fig.suptitle("Expected "+title+" conditioned cumlative returns plotted since market open for "+key)
        plt.tight_layout()
        plt.show()

def returns_discussion(ticker,type: dict,data: dict):

    count = 0
    store = pd.DataFrame(columns=["Year", "Month", "Hour","Max_Return", "Key"])
    hold={}

    for key in ticker:

        condition_count=0
        cond_data=condition_data(data, type)[key]
        cond_data["Date"] = cond_data.index.date
        unique_days = cond_data.index.to_series().dt.date.unique()
        nunique_days=len(unique_days)

        for day in unique_days:

            temp=pd.DataFrame()
            mask = np.isin(cond_data["Date"], day)
            temp[day] = pd.DataFrame(cond_data[mask][key])

            if (temp[day].values>0).sum()>=3:

                condition_count+=1
                count+=1
                year=day.year
                month=day.month
                max=temp[day].values.max()
                hour=temp[day].idxmax().hour
                add={"Year": year,"Month": month,"Hour": hour,"Max_Return": max,"Key": key}
                store.loc[count]=add

        hold[key]=condition_count/nunique_days
    hold = pd.DataFrame(list(hold.items()), columns=['Stock', 'Percentage'])

    return store,hold


if __name__== "__main__":

    ADR_tickers=['BP','BTI','DEO','SHEL', 'GSK','HSBC', 'UL',   'RIO',  'RELX', 'NGG']
    EUR_tickers = ["BP.L", "BATS.L", "DGE.L", "SHEL.L","GSK.L", "HSBA.L", "ULVR.L","RIO.L","REL.L","NG.L",]
    ADR_data = get_intraday_returns(ADR_tickers)
    SP_data = get_intraday_returns("SPY")
    EUR_data = get_hourly_returns(EUR_tickers)
    FT_data = get_hourly_returns(["^FTLC"])
    EUR_data_adjusted = adjust_hourly_data(EUR_data, FT_data)
    ADR_data_adjusted = adjust_data(ADR_data, SP_data)


    #plots of the ADR returns' z_scores as well as their disitrbutions (showing that they are normal)
    #z_scores_plot(ADR_data_adjusted, num_sigma=1.5)

    positive, negative, neutral = event_class(ADR_data_adjusted, num_sigma=1.5)
    #plot_expected_cum(EUR_data_adjusted)

    print("\nNumber of Positive, Neutral and Negative intraday ADR returns \n")
    print(z_scores_statistics(ADR_data_adjusted,num_sigma=1.5),"\n")

    #plots the days that are positive/negative for the market morning session
    #note to plot the returns need to uncomment the code below, change type to positive/negative
    #for i in EUR_tickers:
    #    plot_day(i,type=positive,data=EUR_data_adjusted,title="positive")


    store,hold=returns_discussion(EUR_tickers,data=EUR_data_adjusted,type=negative)
    print("\n",hold)
    print("Average percentage: {:.2f}".format(np.mean(hold["Percentage"])))

    data_2021=store[store["Year"]==2021]
    data_2022=store[store["Year"]==2022]
    data_2023=store[store["Year"]==2023]

    for title,data in zip(["2021","2022","2023"],[data_2021,data_2022,data_2023]):

        plt.hist(data["Month"],bins=12)
        plt.xlabel("Month")
        plt.title("Conditioned month distribution for "+title)
        plt.xticks(np.arange(1,13,1))
        plt.ylabel("Frequency")
        plt.show()

    hour_freq=store["Hour"].value_counts().reset_index()
    hour_freq.columns=["Hour","Frequency"]
    print("\nFrequnecy of hour at maximum return for more than 2 positive returns in a day \n",hour_freq)


#####################################################################################################################
#Due to the limitations of yfinance, I can only download hourly data for the last two years from today. Hence my data
#range is from 2021-2023. In this range, I calculated the ADR intraday returns and found that they were normally distributed
#which is what we should expect. Because of this distribution, I classified abnormalities using the standard deviation and mean
#which charachterise a normal distribution. I took outliers to be +/- 1.5 sigma from the mean. This meant that around 87% of
#the days were classified as neutral and ~13% were positive or negative. I then conditioned the EUR morning session specific
#cumulative returns (for UK stocks only so I could use the FTSE 350 consistently), and plotted the specific cumulative returns
#for the European morning market. I plotted the individual days, which were between 14-39 days for the positive/negative conditions
#Having done this, I further checked on how many of these days there were more than 2 positive cumulative returns. This was a
#proxy to if the conditioned day had an overall positive return. I also looked at the distribution of months to see if particular
#months had more conditioned days. I also looked to see at what hour the maximum return occurred at.

#looking at 10 ADRS on the NYSE with their 10 underlying stocks in the UK I found that on positive|negative conditioned days, 41%|47% of those
#days had overall positive returns. Since we are looking at specific returns, fluctuations due to the respective American and European markets
#are not reflected in the adjusted data. Hence, we would expect changes in the returns to be due to charachteristics of the underlying stocks
#themselves rather than general market momemntum. It is important to note that the European market morning closes before the US market opens. From my analysis, a positive
#intraday return in America did not necessarily imply that the returns in Europe had been positive too. In fact, negative returns in America
#often happened after positive moves in Europe. This implies that using the underlying stock data as a proxy for the performance of the ADR
#stock is not reliable. I further looked at the hours at which the maximum returns in conditioned days happened, to check whether they would be
#within the first 2 hours, where often more volume is traded. This was not the case on whole, as I found that the distribution for these hours
#slightly favoured 10 and 12 oclock.








