import numpy as np
import matplotlib.pyplot as py
from PIL import Image
from binarytree import Node
import sys
import itertools

def main():
    # machine learning pog
    test_image = Image.open('Leaves_Masked.jpg')
    image_completion(test_image)
    test_image = Image.open('Wood_Masked.jpg')
    image_completion(test_image)
    
    
def image_completion(test_image):
    # simplify data
    test_data = np.array(test_image)
    test_data = (test_data//50)
    
    cp = itertools.product(range(6),repeat=3)
    colors = list(p for p in cp)
    colors = np.array(colors)

    new_data = np.empty([0, 900], dtype=int)
    for i in range(900):
        inds = test_data[i] == colors[:, None]
        row_sums = inds.sum(axis = 2)
        j, k = np.where(row_sums == 3) # Check which rows match in all 3 columns

        pos = np.ones(test_data[i].shape[0], dtype = 'int') * -1
        pos[k] = j
        new_data = np.append(new_data, [pos], axis = 0)

    count = 0
    y = np.empty(10000, dtype = int)
    x = np.empty([3, 10000], dtype = int)
    x[0] = np.full(10000, 1)
    for j in range(1, 101):
        for k in range(1, 101):
            y[count] = new_data[j][k]
            x[1][count] = new_data[j - 1][k]
            x[2][count] = new_data[j][k - 1]
            count += 1
    
    w = [[0, 0, 0]]
    # run logistic regression

    vals = log_regression(x, y, w, 10000)

    # loop in a spiral to fill in mask
    k = 300
    l = 300
    m = 601
    n = 601
    while (k < m and l < n):
        # Print the first row from the remaining rows
        for i in range(l, n):
            test_x = np.array([1, new_data[k - 1][i], new_data[k][i - 1]])
            new_data[k][i] = int(np.dot(vals, test_x))
        k += 1
 
        # Print the last column from the remaining columns
        for i in range(k, m):
            test_x = np.array([1, new_data[i - 1][n - 1], new_data[i][n]])
            new_data[i][n-1] = int(np.dot(vals, test_x))
        n -= 1
 
        # Print the last row from the remaining rows
        if (k < m):
            for i in range(n - 1, (l - 1), -1):
                test_x = np.array([1, new_data[m][i], new_data[m - 1][i + 1]])
                new_data[m - 1][i] = int(np.dot(vals, test_x))
            m -= 1
 
        # Print the first column from the remaining columns
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                test_x = np.array([1, new_data[i + 1][l], new_data[i][l - 1]])
                new_data[i][l] = int(np.dot(vals, test_x))
            l += 1
    
    # Convert data back into RGB
    final_data = np.empty([900, 900, 3])
    for i in range(900):
        for j in range(900):
            final_data[i][j] = colors[new_data[i][j]]
    final_data = final_data*50


    # create image from numpy array
    PIL_image = Image.fromarray(np.uint8(final_data)).convert('RGB')
    py.imshow(PIL_image)
    py.show()

def log_regression(x, y, w, size):
    for t in range(1, 10000 + 1):
        i = np.random.randint(size)
        alpha = 1/t
 
        w_t_plus = np.array([])
        for j in range(3):
            w_t_plus = np.append(w_t_plus, w[t - 1][j] - alpha*(np.dot(w[t - 1], x)[i] - y[i]) * x[j][i])
 
        w = np.concatenate((w, [w_t_plus]))

        sum = 0
        fx = 1/(1 + pow(np.e, -np.dot(w_t_plus, x)))
        for i in range(size):
            if fx[i] == 1:
                continue
            sum += -(y[i]*np.log(fx[i])) - (1-y[i])*np.log(1-fx[i])
        err = 1/size * sum
        
        if t > 51 and np.abs(plot_y[plot_y.size - 1] - plot_y[plot_y.size  - 50]) <= 0.0001:
            break
    print(w[t])
    return w[t]


if __name__ == "__main__":   
    main()