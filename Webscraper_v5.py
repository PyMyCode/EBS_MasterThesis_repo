# -*- coding: utf-8 -*-


"""
Created on Fri Jun 12 16:26:28 2020
Notes:
    1. Manually set the sorting of on website such that the newest ones come first
    2.

@author: 20044
"""
import bs4 as bs
'With this module - or its request function - we open the URL, the content of which we want to save in a BeautifulSoup object'
import urllib.request
'With this module, a running program can be interrupted for any time.'
import time
'provide the collected data with timestamps'
from datetime import datetime
import pandas as pd
import json
'access the folder structure in Windows.'
from urllib.request import Request, urlopen

###########################INPUT####################################
max_time = 60000
last_page_no = 1700
###########################INPUT####################################

'Measures the time in seconds from 01/01/1970 until now. Used to calculcate runtime'
start_time = time.time()
  
'A counter is defined that stores the number of runs completed.'
count = 1

page_no = 0

'Loop defined on the bases of runtime'
while (time.time() - start_time) < max_time:
    
    start = start = time.time()
    
    print("Loop " + str(count) + " startet.")

    df = pd.DataFrame()
    
    'Initiates an empty list that will later be populated with URLs'
    l=[]
    
    #looping through the pages
    
    
    
    try:
        
        'Getting the entire source code into a variable'
        
        page_no +=1 
        
        url_str = "https://www.immobilienscout24.de/Suche/de/hessen/frankfurt-am-main/wohnung-mieten?pagenumber=" + str(page_no)
        
        print(url_str)
        
        soup = bs.BeautifulSoup(urllib.request.urlopen(url_str).read(),'lxml')        
        
        'checking for <a> tags'
        'Getting /expose/119530516 from the string'
        'saving it in l'
        for paragraph in soup.find_all("a"):

            if r"/expose/" in str(paragraph.get("href")):
                l.append(paragraph.get("href").split("#")[0])

            l = list(set(l))

        'opening each property'
        for item in l:

            try:
                "visiting each URL from l list"
                soup = bs.BeautifulSoup(urllib.request.urlopen('https://www.immobilienscout24.de'+item).read(),'lxml')
                
                'source code is searched for <script> tags'
                'searches for the word “keyValues”'
                
                help_str = str(soup.find_all("script")).split("keyValues = ")[1].split("}")[0]+str("}")
                
                data = pd.DataFrame(json.loads(help_str),index=[str(datetime.now())])

                data["URL"] = str(item)
                
                beschreibung = []

                for i in soup.find_all("pre"):
                    
                    beschreibung.append(i.text)

                data["beschreibung"] = str(beschreibung)

                df = df.append(data)
                
                'removing the listing that caused removed'

            except Exception as e: 
                print(str(datetime.now())+": " + str(e))
                l = list(filter(lambda x: x != item, l))
                print("ID " + str(item) + " entfernt.")
             

        print("Page " + str(page_no) + " endet.")
               
        
        filepath = "C:\\Users\\20044\\.spyder-py3\\Practice\\Webscraping\\Raw Data\\"
        
           
        df.to_csv(filepath+str(datetime.now())[:19].replace(":","").replace(".","")+".csv",sep=";",decimal=",",encoding='utf-8-sig',index_label="timestamp")
        
        print("Loop " + str(count) + " endet.")
        
        count+=1       
        
        
    except Exception as e: 
        print(str(datetime.now())+": " + str(e))
    
    
    'stopping at a certain number of pages'
    if page_no == last_page_no:
        break
    else:
        pass
    
    'calculating the looping time'
    
    print("total time taken this loop: ", time.time() - start)
           

print("Last Page number is " + str(page_no))

print("FERTIG!")


