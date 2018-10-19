import requests as Req
import re
from bs4 import BeautifulSoup
import random as rnd
from typing import List


MAIN_LINK = "https://www.taquitos.net"
HEADER ={'User-Agent': 'Mozilla/5.0'}
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
    soup = BeautifulSoup(Req.get(url, headers=HEADER).content, 'html.parser', headers=HEADER)
    snack_review = soup.find_all(id="reviewtop")
    assert len(snack_review) == 1, "Wrong number of elements in reviewtop div"
    snack_review = snack_review[0]
    s_title = soup.find('title').get_text()
    s_company = None;
    try:
        for el in snack_review.find_all("p", class_="detail"):
            # TODO: We could also get the review text and images from here!
            # Indicates what that paragraph contains
            first_bold = el.find_all('b')[0].get_text()
            if 'Company' in first_bold:
                 s_company = el.find_all('a')[0].get_text()
    except:
        print("Error while getting snack review content")

    new_snack = Snack(s_title)
    if s_company:
        new_snack.company = s_company
    if country:
        new_snack.country = country
    if category:
        new_snack.category = category
    trunc = s_title[:30]+"(...)" if len(s_title) > 33 else s_title
    print(f"\tCreated {trunc}")
    return new_snack

def scrap_snack_list(url:str, country:str=None) -> List:
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
    country_link = "https://www.taquitos.net/snacks-by-country/"
    soup = BeautifulSoup(Req.get(country_link, headers=HEADER).content, 'html.parser')
    all_snacks = []
    print(soup)
    for el in soup.find('div', class_='triple').find_all('li'):
        text = el.get_text()
        link = el.find('a')
        # If it's just one snack, the site doesn't redirect to a snacklist page
        snack_qtd = re.search(r'(\(.+\))', text).group()
        assert snack_qtd
        snack_qtd = int("".join(filter(str.isdigit, snack_qtd)))
        rel_link = link['href']
        country = link.get_text().strip()
        if snack_qtd == 1:
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

if __name__ == '__main__':
    main()