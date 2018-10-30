import os
import re
import time
import json
import string
from urllib.request import urlretrieve
import random as rnd
import requests as Req
from typing import List
from bs4 import BeautifulSoup
#from proxy_manager import ProxyManager


MAIN_LINK = "https://www.taquitos.net"
HEADER = {'User-Agent': 'Mozilla/5.0'}
PROXY = {
  'http': 'http://10.10.1.10:3128',
  'https': 'http://10.10.1.10:1080',
}
DELAY = 5 # Delay the scrapping so we don't get blocked
# List of user agents to rotate
user_agent_list = [
   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    #Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)']

def update_global_proxy(proxy:str):
    global  PROXY
    PROXY['http'] = f"http://{proxy}"
    PROXY['https'] = f"http://{proxy}"


class Snack(object):
    def __init__(self, title):
        self.title = title;

    def __str__(self):
        own_objects = self.__dict__
        snack_str = "{\n"
        for attr, val in own_objects.items():
            snack_str += f"    {attr} : {val}\n"
        snack_str += "}"
        return snack_str

def create_snack_from_url(url:str, country:str=None, category:str=None):
    '''
    Given an _INDIVIDUAL_ snack URL, obtain the info from said snack
    and create a new Snack object representing the newly scrapped snack,
    returning the individual Snack object.

    It's the most deeper depth method in the scrapping routine
    '''
    global HEADER, PROXY
    time.sleep(DELAY)
    soup = BeautifulSoup(Req.get(url, headers=HEADER).content, 'html.parser')
    snack_review = soup.find_all(id="reviewtop")
    assert len(snack_review) == 1, "Wrong number of elements in reviewtop div"
    snack_review = snack_review[0]
    s_title = soup.find('title').get_text()
    s_company = None;
    s_description = None;
    printable = set(string.printable) # valid chars for a filename
    img_links = []
    extra_info = "" # String to store extra information to print later
    try:
        # Extract textual information
        for el in snack_review.find_all("p", class_="detail"):
            # Indicates what that paragraph contains
            first_bold = el.find_all('b')[0].get_text()
            if 'Company' in first_bold:
                 s_company = el.find_all('a')[0].get_text()
            if "From the package" in el.get_text():
                children = list(el.children)
                if len(children) > 1:
                    s_description = children[1].strip()
        # Extract all image links from the review
        for img in snack_review.find_all("img", class_ = "reviewpictall"):
            img_links.append(MAIN_LINK+img.get("src"))
    except:
        print("Error while getting snack review content")

    new_snack = Snack(s_title)
    if s_company:
        new_snack.company = s_company
    if country:
        new_snack.country = country
    if category:
        new_snack.category = category
    if s_description:
        new_snack.description = s_description
    if img_links:
        foldername = new_snack.title.strip().replace(" ","_").lower()
        new_snack.folder_name = "".join(list(filter(lambda x: x in printable, foldername)))
        assert new_snack.folder_name != None
        if not os.path.exists("image/"):
            os.makedirs("image/")
        if not os.path.exists("image/"+new_snack.folder_name):
            os.makedirs("image/"+new_snack.folder_name)
        for link in img_links:
            img_name = link.split('/')[-1]
            urlretrieve(link, "image/"+new_snack.folder_name+f"/{img_name}")
            extra_info += f"\t\tsaved image: {img_name}\n"
    # Truncate the title
    trunc = s_title[:30]+"(...)" if len(s_title) > 33 else s_title
    print(f"\tCreated {trunc}")
    print(extra_info)
    return new_snack

def scrap_snack_list(url:str, country:str=None) -> List:
    global HEADER, PROXY
    time.sleep(DELAY)
    HEADER = {'User-Agent': rnd.choice(user_agent_list)}
    soup = BeautifulSoup(Req.get(url,headers=HEADER).content, 'html.parser')
    snacks_from_list = []
    list_element = []
    try:
        list_element = soup.find('div', id='longlist').find('ul').find_all('a')
    except:
        print(soup.prettify())
        exit()
    # Traverse the list and get every link
    for link in list_element:
        try:
            rel_link = link['href']
            snacks_from_list.append(create_snack_from_url(MAIN_LINK+rel_link, country=country))
        except:
            ValueError("Error while scraping snack list")
    return snacks_from_list

def scrap_per_country() -> List:
    global HEADER, PROXY
    country_link = "https://www.taquitos.net/snacks-by-country/"
    """ proxy_scrape = ProxyManager(country_link,
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36") """
    time.sleep(DELAY)
    HEADER = {'User-Agent': rnd.choice(user_agent_list)}
    """ proxy_scrape.refresh_proxy_status()
    proxy_scrape.update_proxy_list()
    proxy = proxy_scrape.get_proxy()
    update_global_proxy(proxy) """
    soup = BeautifulSoup(Req.get(country_link, headers=HEADER).content, 'html.parser')
    all_snacks = []
    #print(soup)
    # TODO: You can change the quantity of countries to scrap in the line below, by changing the array
    for el in soup.find('div', class_='triple').find_all('li')[:9]:
        # Get a proxy
        """ print("Getting new proxy...")
        proxy = proxy_scrape.get_proxy()
        while proxy == None:
            proxy_scrape.refresh_proxy_status()
            proxy_scrape.update_proxy_list()
            proxy = proxy_scrape.get_proxy()
        update_global_proxy(proxy) """
        text = el.get_text()
        link = el.find('a')
        # If it's just one snack, the site doesn't redirect to a snacklist page
        snack_qtd = re.search(r'(\(.+\))', text).group()
        assert snack_qtd
        snack_qtd = int("".join(filter(str.isdigit, snack_qtd)))
        rel_link = link['href']
        country = link.get_text().strip()
        if snack_qtd == 1:
            HEADER = {'User-Agent': rnd.choice(user_agent_list)}
            all_snacks.append(create_snack_from_url(url=MAIN_LINK+rel_link, country=country))
            continue
        # Create from the whole country
        try:
            # If it's a country, it doesn't have 'http' in the link
            if 'http' not in rel_link:
                print(f"Scrapping from {country}: ")
                all_snacks += scrap_snack_list(url=MAIN_LINK+rel_link, country=country)
        except:
            raise ValueError("Error while scraping countries")
    return all_snacks


def main():
    snack_list = scrap_per_country()
    print(f"\n\n FINISH: {(len(snack_list))} Snacks created! \n Here's some of them:")
    
    if len(snack_list) >= 10:
        start = rnd.randint(0, len(snack_list) - 6)
        for snack in snack_list[start:start + 5]:
            print(snack)
    else:
        for snack in snack_list:
            print(snack)
    data = json.dumps(list(map(lambda obj: obj.__dict__, snack_list)))
    with open('snacks.json', 'w') as outfile:
    	json.dump(data, outfile)

if __name__ == '__main__':
    main()
