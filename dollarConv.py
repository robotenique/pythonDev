import requests
import json
# Simple aplication for calculating the monthly based sallary of USD in BRL


def main():
    # JSON REQUEST
    r = requests.get('http://api.fixer.io/latest?base=USD')
    report = r.json()

    if r.status_code == 200:
        exRate = float(report['rates']['BRL'])
        nsd = float(input("USD annual sallary : $"))
        nsd *= exRate
        nsd /= 12
        print("Calculated monthly sallary in BRL: R$ ", round(nsd, 2))

    else:
        print("Failed to connect to Currency rate server. Try again later.")
        quit()


if __name__ == '__main__':
    main()
