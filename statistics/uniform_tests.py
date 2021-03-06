import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt

""" Statistics tests to check if a dataset is uniformly generated,
    using buckets and the chisquare test.
"""


def gen_buckets(num_buckets, data, max_val=256):
    """ Generate multiple buckets from randomly shuffled parts of the data.
        This has only been tested for positive numbers >:D

    Arguments:
        num_buckets {int} -- Number of bucket lists to generate
        data {int} -- array of integers with the data

    Keyword Arguments:
        max_val {int} -- the max value in the data, used to generate the length of the buckets lists (default: {256})

    Returns:
        list -- list of all the buckets lists generated, each bucket list has 'max-val' of length
    """

    default_size_of_bucket = int(len(data)/3)
    print(f"Bucket size: {default_size_of_bucket}")
    all_buckets = []
    for i in range(num_buckets):
        curr_buck = [0 for _ in range(max_val)]
        np.random.shuffle(data)
        curr_sample = data[0:default_size_of_bucket]
        for i in range(len(curr_sample)):
            curr_buck[curr_sample[i]] += 1
        all_buckets.append(curr_buck)
    return all_buckets

def hamming_weight(num):
    """Calculate the hamming weight of a number

    Arguments:
        num {integer (positive)} -- the number to calculate the hamming weight

    Returns:
        int -- number of 1's in the binary form of the number
    """

    return bin(num).count("1");

def main():
    # Load data from text
    data = np.loadtxt("res_with_fair_bit.txt", dtype=int)
    """ Calculate 1000x buckets from the same length of
        randomly shuffled data, then calculate the chisquare test
        for each pvalue. This can be used to test if the said dataset
        came from a Uniform distribution or not. Then, I create a histogram
        of pvalues to test the distribution.
    """
    num_buckets_init = 1000
    my_vals = []
    for bckt in gen_buckets(num_buckets_init, data):
        my_vals.append(np.around(chisquare(bckt).pvalue, decimals=5))
    # If the pvalues have a 'uniform' distribution over [0, 1], we don't negate the null hypothesis
    plt.hist(my_vals)
    plt.show()

    """ Calculate the hamming weight (number of '1's in binary form)
        of the whole data, then create buckets for a random sample of
        randomly shuffled data. Then apply the chisquare test to check
        if the data came from a uniform distribution or not.
    """
    num_buckets_hweight = 1000
    hweight = np.vectorize(hamming_weight)
    data_hamming = hweight(data)
    my_vals = []
    """ I know the data is one byte, this means that if max_val = 9, 
    	each bucket will be related to these indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]"""
    for bckt in gen_buckets(num_buckets_hweight, data_hamming, max_val=9):
        #plt.plot(bckt) # You can see that the data is a gaussian / binomial distribution
        my_vals.append(np.around(chisquare(bckt).pvalue, decimals=5))
    # If the pvalues are very close to zero, we don't accept the null hypothesis => High probability of NOT being a uniform distribution
    plt.hist(my_vals)
    plt.show()


initfilepath = "dakarand-rng"

engine_data = {'carbon': ['carbon/fair_bit_carbon.txt',
  'carbon/no_fair_bit_carbon.txt',],
 'chakra': ['chakra/fair_bit_chakra.txt',
  'chakra/no_fair_bit_chakra.txt'],
 'firefox': ['firefox/no_fair_bit_firefox.txt',
  'firefox/fair_bit_firefox.txt'],
 'node11': ['node11/no_fair_bit_node11.txt',
  'node11/fair_bit_node11.txt']}
""" engine_data = {'carbon': ['carbon/fair_bit_carbon.txt',
  'carbon/no_fair_bit_carbon.txt']} """

def check_uniformity(data_set, plot_axis, title, color="blue"):
    num_buckets_init = 1000
    my_vals = []
    for bckt in gen_buckets(num_buckets_init, data_set):
        my_vals.append(np.around(chisquare(bckt).pvalue, decimals=5))
    # If the pvalues have a 'uniform' distribution over [0, 1], we don't negate the null hypothesis
    plot_axis.hist(my_vals, color=color)
    plot_axis.set_title(title)

def project_hamming_weight(data_set, plot_axis, title, color="red"):
    # No buckets here, just plain histogram to check VISUALLY the distribution
    num_buckets_hweight = 1000
    hweight = np.vectorize(hamming_weight)
    data_hamming = hweight(data_set)
    plot_axis.hist(data_hamming, color=color, bins="auto")
    plot_axis.set_title(title)

f1, plot_axis = plt.subplots(4, 2, sharex=True)
f2, plt_hamming_weight = plt.subplots(4, 2, sharex=True)
for engine, row in zip(engine_data.keys(), range(len(engine_data.keys()))):
    print(f"----------- {engine} -----------")
    for data_set_name, column in zip(engine_data[engine], range(len(engine_data[engine]))):
        print(f"{' '*column}Processing {data_set_name}:")
        check_uniformity(np.loadtxt(initfilepath+"/"+data_set_name, dtype=int), plot_axis[row][column], title=data_set_name)
        project_hamming_weight(np.loadtxt(initfilepath+"/"+data_set_name, dtype=int), plt_hamming_weight[row][column], title=data_set_name)
f1.tight_layout()
f2.tight_layout()
plt.show()

""" for engine, row in zip(engine_data.keys(), range(len(engine_data.keys()))):
    f, plot_axis = plt.subplots(4, 2, sharex="col", sharey="row")
    print(f"----------- {engine} -----------")
    for data_set_name, column in zip(engine_data[engine], range(len(engine_data[engine]))):
        print(f"{' '*column}Processing {data_set_name}:")
        check_dist_hamming_weight(np.loadtxt(initfilepath+"/"+data_set_name, dtype=int), plot_axis[row][column], title=data_set_name, color="red")
plt.show() """



"""
if __name__ == '__main__':
    main()  """
