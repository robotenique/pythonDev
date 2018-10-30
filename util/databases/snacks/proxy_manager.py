import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup


def freeproxylist(user_agent):
    proxies = {}
    response = requests.get('https://www.free-proxy-list.net/', headers={'User-Agent': user_agent}, timeout=(9, 27))
    soup = BeautifulSoup(response.text, 'html.parser')
    proxy_list = soup.select('table#proxylisttable tr')
    for p in proxy_list:
        info = p.find_all('td')
        if len(info):
            proxy = ':'.join([info[0].text, info[1].text])
            if info[6].text == 'yes':
                proxies.update({proxy: {'country_code': info[2].text, 'country': info[3].text, 'privacy': info[4].text,
                                    'google': info[5].text, 'https': info[6].text, 'last_checked': None,
                                    'alive': True}})
    return proxies


class ProxyManager:

    def __init__(self, test_url, user_agent):
        self.test_url = test_url
        self.user_agent = user_agent
        self.thread_pool = ThreadPoolExecutor(max_workers=50)
        self.proxies = {}
        self.update_proxy_list()

    def update_proxy_list(self):
        try:
            self.proxies = freeproxylist(self.user_agent)
        except Exception as e:
            logging.error('Unable to update proxy list, exception : {}'.format(e))

    def __check_proxy_status(self, proxy, info):
        info['last_checked'] = datetime.datetime.now()
        try:
            res = requests.get(self.test_url, proxies={'https': proxy}, timeout=(3, 6))
            print(res.content)
            res.raise_for_status()
        except Exception as e:
            info['alive'] = False
        else:
            info['alive'] = True
        return {proxy: info}

    def refresh_proxy_status(self):
        results = [self.thread_pool.submit(self.__check_proxy_status, k, v) for k, v in self.proxies.items()]
        for res in results:
            result = res.result()
            self.proxies.update(result)

    def get_proxies_key_value(self, key, value):
        proxies = []
        for k, v in self.proxies.items():
            match = v.get(key)
            if match == value:
                proxies.append(k)
        return proxies

    def get_proxy(self):
        proxy = None
        for k, v in self.proxies.items():
            alive = v.get('alive')
            if alive:
                return k
        return proxy

"""# Create an instance of our proxy manager
proxy_scrape = ProxyManager("https://www.reddit.com",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36")

# Refresh the status of the proxies we pulled on initialization
proxy_scrape.refresh_proxy_status()
# Get a single working proxy
proxy = proxy_scrape.get_proxy()
times_made = 0
while proxy == None and times_made < 1e2:
    proxy_scrape.update_proxy_list()
    proxy = proxy_scrape.get_proxy()
    times_made += 1
link_proxy = f"https://{proxy}"
proxies = {
    'https': link_proxy
}
print(proxy)
res = requests.get("https://www.reddit.com", proxies=proxies)
print(res)
print(res.content)
proxy_scrape.update_proxy_list()
for k, v in proxy_scrape.proxies.items():
    alive = v.get('alive')
    if alive:
        print(k)
 """

# Make a fresh scrape of free-proxy-list.net
# proxy_scrape.update_proxy_list()