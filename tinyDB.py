from tinydb import TinyDB, Query
import time
import datetime
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
db = TinyDB("zika.json")
t1 = db.table("central")
db.purge()
Fruit = Query()
print(t1.search(Fruit.type == 'apple')[0]['type'])
t2 = db.table("fisica")
t3 = db.table("quimica")
t4 = db.table("prefeitura")
t5 = db.table("refeicoes")

print(db.all())
