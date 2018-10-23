import requests
import traceback
from itertools import cycle
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

MAX_PROXIES_NUM = 10
TIMEOUT = 1.5
def get_proxies():
    ua = UserAgent()
    proxies = []
    url = 'https://www.sslproxies.org/'

    response = requests.get(url, headers={"user-agent" : ua.random})
    soup = BeautifulSoup(response.content, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')

    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
            'ip':   row.find_all('td')[0].string,
            'port': row.find_all('td')[1].string
        })
    print(f"== Scrapped {len(proxies)} proxies in total ==")
    #proxy_index = random_proxy()
    #proxy = proxies[proxy_index]
    #for n in range(1, 100):
    #	req = Request('http://icanhazip.com')
    #	req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
    return proxies

def random_proxy():
  return random.randint(0, len(proxies) - 1)

#If you are copy pasting proxy ips, put in the list below
#proxies = ['177.135.236.122:48795']
proxies = get_proxies()
proxy_pool = cycle(proxies)

url = 'https://httpbin.org/ip'

for i in range(1,MAX_PROXIES_NUM):
    #Get a proxy from the pool
    proxy = next(proxy_pool)
    print("Request #%d"%i)
    good_proxies =  []
    try:
        curr_proxy = f"https://{proxy['ip']}:{proxy['port']}"
        response = requests.get(url,proxies={"http": curr_proxy, "https": curr_proxy}, timeout=TIMEOUT)
        print(f"Success: {response.json()}")
        good_proxies.append({"http": curr_proxy, "https": curr_proxy})
    except:
        #Most free proxies will often get connection errors. You will have retry the entire request using another proxy to work.
        #We will just skip retries as its beyond the scope of this tutorial and we are only downloading a single url
        print("Skipping. Connnection error or Timeout")
print(good_proxies)
for prox in good_proxies:
    print(prox)