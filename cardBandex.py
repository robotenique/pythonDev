#!/usr/bin/env python3
import subprocess as sb
import html
import re
import time
from datetime import datetime as dt
import datetime
import sys
try:
    from tinydb import TinyDB, Query
    def execute_query(cardapio, code):
        days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        rests = ["fisica", "quimica", "prefeitura", "central"]
        ts = time.time()
        ts = dt.fromtimestamp(ts).strftime("%d-%m-%Y")
        dict_res = {key : value for (key, value) in zip([8, 9, 7, 6], rests)}
        refeicao = cardapio[days[dt.today().weekday()]]
        weekday = days[dt.today().weekday()].lower()
        db = TinyDB("cardapio.json")
        ingr_table = db.table("ingrediente")
        rest_table = db.table(dict_res[code])
        almoco = generate_refeicao(refeicao[0].strip().split("\n"), ingr_table)
        jantar = generate_refeicao(refeicao[1].strip().split("\n"), ingr_table)
        almoco = almoco if almoco else False
        jantar = jantar if jantar else False
        if almoco or jantar:
            rest_table.insert({"data" : ts, "diasemana" : weekday,
                               "almoco" : almoco, "jantar" : jantar})

    def generate_refeicao(list_i, table):
        list_i = list(filter(None, list_i))
        if(len(list_i) < 2): return None
        dict_keys = ["comum", "principal", "opcao", "acomp", "sobremesa", "adicional"]
        dict_refeicao = dict()
        for (key, value) in zip(dict_keys, list_i):
            dict_refeicao[key] = value.replace("Opção:","").strip().lower()
        queryS = Query()
        for ingr in list_i:
            ingr = ingr.strip().lower()
            query_result = table.search(queryS.tipo == ingr)
            if(len(query_result) == 0):
                table.insert({"tipo" : ingr , "qtd" : 1})
            else:
                preCount = query_result[0]["qtd"]
                table.update({"qtd" : preCount + 1}, queryS.tipo == ingr)
        return dict_refeicao
except ImportError:
    pass

# Global variables
WHITE = "\033[38;5;7m"
GRAY = "\033[38;5;252m"
bCodes = {"Física" :8, "Química" :9, "Prefeitura":7, "Central" :6}
colors  = {8 :"\033[38;5;9m", 9 :"\033[38;5;120m", 7 :"\033[38;5;220m", 6 :"\033[38;5;190m"}
godlike = True
'''
List with featured FOODS! For each item in the list, the first
item is the featured one, and all the others followed by a ","
are a string to NOT match.
For example, if you don't like "pudim de abacate" you put "pudim, abacate",
so then it won't match if the word contains "pudim", but not "abacate"!
'''
featured = ["mel", "queijo", "doce, batata", "pudim", "flan", "sugo, berinjela",
            "batata, doce", "abacaxi", "nhoque", "mousse", "chocolate", "estrogonofe",
            "brigadeiro", "milanesa", "pizzaiolo", "fantasia", "mandioca","sorvete", "fricassé", "almôndegas",
            "madeira"]
def get_command(rID):
    cmd = ["curl", "-sw", "-H", "\"Host:uspdigital.usp.br\nConnection:keep-alive\nContent-Length:280\nOrigin:https://uspdigital.usp.br\nUser-Agent:Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Mobile Safari/537.36\nContent-Type:text/plain\nAccept:*/*\nReferer:https://uspdigital.usp.br/rucard/Jsp/cardapioSAS.jsp\nAccept-Encoding:gzip, deflate, br\nAccept-Language:pt-BR,pt;q=0.8,en-US;q=0.6,en;q=0.4,fr;q=0.2\nRequest Payload:\"", "-X", "POST", "-d","callCount=1\nwindowName=1\nnextReverseAjaxIndex=0\nc0-scriptName=CardapioControleDWR\nc0-methodName=obterCardapioRestUSP\nc0-id=0\nc0-param0=string:{}\nbatchId=1\ninstanceId=0\npage=%2Frucard%2FJsp%2FcardapioSAS.jsp%3Fcodrtn%3D0\nscriptSessionId=".format(rID), "https://uspdigital.usp.br/rucard/dwr/call/plaincall/CardapioControleDWR.obterCardapioRestUSP.dwr"]
    return cmd

def parse_cardapio(script_res):
    r = str(script_res).split("\\n")
    r = list(filter(lambda s: "dwr.engine.remote." in s, r))
    r = html.unescape(r[0]).split("[")[1].encode().decode('unicode_escape')
    r = list(filter(None, clean_str(r.replace("\/",", ")).split("{")))
    for i in range(len(r)):
        k = list(filter(None, r[i].split("cdpdia:")))
        k = list(filter(None, "".join(k).split("\"")))
        r[i] = k[0].encode().decode('unicode_escape') if k[0] != " " else "Fechado"
    r = list(zip(r[::2], r[1::2]))
    return r

def clean_str(r):
    d = {"<br>":"\n", "}":""}
    regex = re.compile('|'.join(d.keys()))
    return regex.sub(lambda x: d[x.group()], r)

def create_cardapio(r):
    days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    cardapio = {dia: refeicao for dia, refeicao in zip(days, r)}
    return cardapio

def app_h(k):
    for f in featured:
        f = [x.strip().lower() for x in f.split(",")]
        if f[0] in k.lower().split(" "):
            if(len(list((x for x in f[1:] if x in k.lower()))) == 0):
                return "\033[4m\033[38;5;165m*"+k.strip()+"*\033[0m"
    return k

def print_refeicao(rf, color):
    uLine = lambda x: "\033[38;5;14m"+x+"\033[0m"+color
    tmp = list(filter(None, rf.split("\n")))
    if(len(tmp) > 2):
        for k, idx in zip(tmp, range(len(tmp))):
            print(WHITE+"➤ "+GRAY, end="")
            if idx == 1 or idx == 5 or idx == 3:
                temp = app_h(k) if godlike else k
                print(uLine(temp))
            else: print(k)
    else:
        print(rf)

def print_day(cardapio, tag, color, code=None, day=""):
    days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    d_Index = datetime.datetime.today().weekday() if not day else day - 1
    refeicao = cardapio[days[d_Index]]
    if tag != "":
        tag = 0 if tag == " (Almoço)" else 1
        print_refeicao(refeicao[tag], GRAY)
    else:
        for i, title in zip(refeicao,["Almoço", "Jantar"]):
            print(colors[code]+format_str(title,"",char="-", sz=20)+color)
            print_refeicao(i, GRAY)

def get_TimeTag():
        now = datetime.datetime.now()
        aInf = now.replace(hour=6, minute=0, second=0, microsecond=0)
        aSup = now.replace(hour=14, minute=0, second=0, microsecond=0)
        jInf = now.replace(hour=14, minute=1, second=0, microsecond=0)
        jSup = now.replace(hour=20, minute=1, second=0, microsecond=0)
        if now > aInf and now < aSup:
            return " (Almoço)"
        elif now > jInf and now < jSup:
            return " (Jantar)"
        return ""

def format_str(name, tag, char="=", sz=50, addIcons=True):
    bdSt = list(name+tag)
    l = True
    while(len(bdSt) < sz):
        if l:   bdSt.append(char)
        else:   bdSt = [char] + bdSt
        l = not l
    if(addIcons):
        bdSt[0] = "#"
        bdSt[-1] = "#"
    # Centralize
    if(len(bdSt) < 50):
        bdSt = [" " for x in range((50 - len(bdSt))//2)] + bdSt
    return "".join(bdSt)

def print_AllBdex(tag, dump=False, day="", logo=True):
    days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    if logo: print_logo()
    if day:
        print("\n\n"+format_str(days[day-1],"",char="~", sz=30))
    for name, code in bCodes.items():
        print("\n"+colors[code]+format_str(name, tag)+GRAY)
        cdp = get_command(str(code))
        cdp = parse_cardapio(sb.check_output(cdp))
        cdp = create_cardapio(cdp)
        if(not dump):
            print_day(cdp, tag, GRAY, code=code, day=day)
        else:
            execute_query(cdp, code)

def print_logo():
    fadeColors = ["\033[38;5;{}m".format(i) for i in range(118, 124)]

    logo = [" ____                     _\n",
              "| __ )   __ _  _ __    __| |  ___ __  __\n",
              "|  _ \  / _` || '_ \  / _` | / _ \\\ \/ /\n",
              "| |_) || (_| || | | || (_| ||  __/ >  <\n",
              "|____/  \__,_||_| |_| \__,_| \___|/_/\_\\\n"]
    for color, line  in zip(fadeColors, logo):
        print(color+line, end="", sep="")

def print_usage(wrong_arg=None):
    err = lambda x: "\033[38;5;160m "+x+"\033[38;5;7m"
    runF = lambda x: "\033[38;5;4m"+x+"\033[38;5;7m"
    flagF = lambda x: "\033[38;5;46m"+x+"\033[38;5;7m"
    if(wrong_arg != None):
        print("😞😞😞 Poxa "+sb.check_output(["whoami"]).decode("utf-8").replace("\n","")+
              "," +err("argumento \""+wrong_arg+"\" é inválido! ")+"😞😞😞")
    print("Uso:\n $ "+runF("bandex")+"\n\t Imprime o cardápio do dia de todos os restau"
          "rantes, com as refeições de acordo com o horário de execução do script;\n"+
          flagF("FLAGS:")+"\n\t"+flagF("-a")+" : Imprime apenas os almoços;"+"\n\t"+
          flagF("-j")+" : Imprime apenas os jantares;"+"\n\t"+flagF("-all")+" : Imprime"
          " todas as refeições do dia;"+"\n\t"+flagF("-d número")+" : Imprime"
          " todas as refeições do dia escolhido (segunda = 1, terça = 2, etc);"+
          "\n\t"+flagF("-E ou --EVERYTHING")+" : Imprime todas as refeições da semana! D: ;"+"\n\t"
          +flagF("-h ou --help")+" : Imprime esta página de informação.\n")
    exit()

def main():
    #print(sb.check_output(["whoami"]))

    if(len(sys.argv) == 1):
        tag = get_TimeTag()
        print_AllBdex(tag)
    elif(len(sys.argv) == 2):
        if(sys.argv[1] == "-a"):
            tag = " (Almoço)"
        elif(sys.argv[1] == "-j"):
            tag = " (Jantar)"
        elif(sys.argv[1] == "-all"):
            tag = ""
        elif(sys.argv[1] == "--EVERYTHING" or sys.argv[1] == "-E"):
            for i in range(1, 8):
                print_AllBdex("", day=i, logo = False)
            exit()
        elif(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print_usage()
        else:
            try:
                k = "".join(open("/etc/naoZOA").readlines())
                if(sys.argv[1] == k):
                    print("ENTROU")
                    print_AllBdex("", dump=True)
                    exit()
            except Exception as e:
                pass
            print_usage(sys.argv[1])
        print_AllBdex(tag)
    elif(len(sys.argv) == 3 and sys.argv[1] == "-d"):
        try:
            day_choosen = int(sys.argv[2])
            if day_choosen in range(1,8):
                print_AllBdex("", day=day_choosen)
        except (TypeError, ValueError):
            print_usage()
    else:
        print_usage()


if __name__ == '__main__':
    main()
