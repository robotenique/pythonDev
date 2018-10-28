import numpy as np
def get_freq(title):
    # Read an array, calculate the frequency and sort by frequency (DESC)
    data = np.loadtxt(title, dtype=int)
    max_val = max(data)
    curr_buck = [0 for _ in range(max_val + 1)]
    for i in range(len(data)):
        curr_buck[data[i]] += 1
    topkek =  []
    for i in range(len(curr_buck)):
        topkek.append([i, curr_buck[i]])
    topkek = np.array(topkek)
    print(f"--------- {title} ---------")
    result = topkek[topkek[:,1].argsort()[::-1]]
    # print the first 25 most frequent numbers in the data
    print(result[:25])

# Using some data as example
get_freq("fair_bit_node11.txt")
get_freq("no_fair_bit_node11.txt")