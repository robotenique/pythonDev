import numpy as np

def main():
    num_test = int(input(""))
    for _ in range(num_test):
        num_students = int(input(""))
        data = np.array([list(map(float, input("").split())) for _ in range(6)], dtype=float)
        bad_data = []
        for row in range(data.shape[0]):
            # Makes the row irrelevant if it has constant values
            if len(set(data[row])) == 1:
                data[row] = np.zeros(len(data[row]))
                data[row][0] = 1
        corr = np.corrcoef(data)
        print(1 + np.argmax(corr[0, 1:]))

if __name__ == '__main__':
    main()
