# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:21:39 2017

@author: Arjun
"""
import bittrex.bittrex as bt
import pandas as pd
import numpy as np
import time, threading, random, datetime

def fetch():
    btr = bt.Bittrex(None,None)
    dict_js = btr.get_market_summaries()
    list_result = dict_js["result"]
    return list_result


def candidate_monitor(cand,ind,volchange):
    key = str(ind)+"_"+cand["MarketName"]+" v+"+str(volchange)+"%"
    if key not in candidates:
        candidates[key] = [("0.0%",cand["Bid"],'0.0s')]
        init_bid = cand["Bid"]
        new_bid = init_bid
        count = 0
        stt = time.clock()
        while(1):
            if count>=5:
                return   
            if(time.clock()>=stt+2):
                temp_bid = fetch()[ind]["Bid"]
                if(new_bid != temp_bid):
                    new_bid = temp_bid
                    changes = round((new_bid/init_bid)*100-100,2)
                    candidates[key].append((str(changes)+"%",new_bid,str(round(time.clock()-stt,1))+"s"))
                    count+=1
                stt = time.clock()
  
              
def pd_pprint(candidates,sz):
    global data
    for k in candidates.copy():
        if len(candidates[k])<sz:
            candidates.pop(k)
    inx = len(candidates[random.choice(list(candidates.keys()))])
    data = pd.DataFrame(candidates,columns = candidates.keys(),index=list(range(inx)))
    return data

    
def profit_stats(candidates,sz,app=False):
    global outcome
    data = pd_pprint(candidates,sz)
    print(data)
    np_zeros = np.zeros(data.shape[1])
    n=[]
    minray = []
    maxray = []
    for i in candidates.keys():
            n.clear()
            for j in range(1,len(data)):
                n.append(float(data[i][j][0][:-2]))
            maxray.append(max(n))
            minray.append(min(n))
    np_maxray = np.array(maxray)
    np_minray = np.array(minray)
    rise_percent = np.mean(np_maxray > np_zeros)*100
    no_change = np.mean(np_maxray == np_zeros)*100
    safe_zone = rise_percent+no_change
    fall_risk = 100-safe_zone
    mean_rise = np.mean(np_maxray[np_maxray > np_zeros])
    mean_fall = np.mean(np_minray[np_minray < np_zeros])
    indextits = ["rise probability","no_change probability","highest peak"
                 ,"lowest dip","mean rise","mean fall","safe zone","fall risk"]
    titvals = [round(rise_percent,2),round(no_change,2),round(max(np_maxray),2)
    ,round(min(minray),2),round(mean_rise,2),round(mean_fall,2),round(safe_zone,2),round(fall_risk,2)]
    titvals = [str(i)+"%" for i in titvals]
    outcome = pd.Series([datetime.date.today()],["Date"])
    outcome = outcome.append(pd.Series(titvals,index = indextits))
    if app:
        write(data,outcome)
        print("\n=written to files=\n")
    print("\ncryptos recorded:",data.shape[1])
    return outcome


def write(data,outcome):
    data.to_csv("raw.csv",sep=" ",mode="a")
    outcome.to_csv("raw.csv",sep=" ",mode="a")
    data.to_csv("data.csv",sep=" ",mode="a")
    outcome.to_csv("outcomes.csv",sep=" ",mode="a")
    

def trds():
    return threading.enumerate()
 
candidates = {}           
coins_q_old = fetch()
change = 0.0
start_time = time.time()
while(time.time() <= start_time+600):
    start = time.clock()
    coins_q_new = fetch()
    for i in range(len(coins_q_old)):
        try:
            change =(coins_q_new[i]["Volume"] / coins_q_old[i]["Volume"])*100-100
        except ZeroDivisionError:
            change = 0.0
        change = round(change,1)
        if change>=1 and change<=5:
            print(coins_q_new[i]["MarketName"],": +",round(change,2),"%")
            t = threading.Thread(target = candidate_monitor,args = (coins_q_new[i],i,change))
  #          t.daemon = True
            t.start()
  #          ^ was candidate_monitor(coins_q_new[i],i)            
    coins_q_old = coins_q_new
    print(round(time.clock()-start,2))