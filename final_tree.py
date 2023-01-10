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
    test_data = (test_data//10)
    
    cp = itertools.product(range(26),repeat=3)
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
    y = np.empty(90000, dtype = int)
    x = np.empty([2, 90000], dtype = int)
    for j in range(1, 301):
        for k in range(1, 301):
            y[count] = new_data[j][k]
            x[0][count] = new_data[j - 1][k]
            x[1][count] = new_data[j][k - 1]
            count += 1
    
    # run decision tree
    root = tree(x, y, 0, 100, 0)
    
    # loop in a spiral to fill in mask
    k = 300
    l = 300
    m = 601
    n = 601
    while (k < m and l < n):
 
        # Print the first row from the remaining rows
        for i in range(l, n):
            ptr = root
            while(1 == 1):
                if isinstance(ptr.value, str):
                    vals = np.array(ptr.value.split(" ", 1))
                    values = np.empty(2)
                    for q in range(vals.size):
                        values[q] = float(vals[q])
                    
                    if values[0] == 1.0:
                        if new_data[k - 1][i] > values[1]:
                            ptr = ptr.right
                        else:
                            ptr = ptr.left
                    else:
                        if new_data[k][i - 1] > values[1]:
                            ptr = ptr.right
                        else:
                            ptr = ptr.left
                else:
                    new_data[k][i] = int(ptr.value)
                    break
 
        k += 1
 
        # Print the last column from the remaining columns
        for i in range(k, m):
            ptr = root
            while(1 == 1):
                if isinstance(ptr.value, str):
                    vals = np.array(ptr.value.split(" ", 1))
                    values = np.empty(2)
                    for q in range(vals.size):
                        values[q] = float(vals[q])
                    
                    if values[0] == 1.0:
                        if new_data[i - 1][n - 1] > values[1]:
                            ptr = ptr.right
                        else:
                            ptr = ptr.left
                    else:
                        if new_data[i][n] > values[1]:
                            ptr = ptr.right
                        else:
                            ptr = ptr.left
                else:
                    new_data[i][n-1] = int(ptr.value)
                    break
        n -= 1
 
        # Print the last row from the remaining rows
        if (k < m):
            for i in range(n - 1, (l - 1), -1):
                ptr = root
                while(1 == 1):
                    if isinstance(ptr.value, str):
                        vals = np.array(ptr.value.split(" ", 1))
                        values = np.empty(2)
                        for q in range(vals.size):
                            values[q] = float(vals[q])
                        
                        if values[0] == 1.0:
                            if new_data[m][i] > values[1]:
                                ptr = ptr.right
                            else:
                                ptr = ptr.left
                        else:
                            if new_data[m - 1][i + 1] > values[1]:
                                ptr = ptr.right
                            else:
                                ptr = ptr.left
                    else:
                        new_data[m - 1][i] = int(ptr.value)
                        break
 
            m -= 1
 
        # Print the first column from the remaining columns
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                ptr = root
                while(1 == 1):
                    if isinstance(ptr.value, str):
                        vals = np.array(ptr.value.split(" ", 1))
                        values = np.empty(2)
                        for q in range(vals.size):
                            values[q] = float(vals[q])
                        
                        if values[0] == 1.0:
                            if new_data[i + 1][l] > values[1]:
                                ptr = ptr.right
                            else:
                                ptr = ptr.left
                        else:
                            if new_data[i][l - 1] > values[1]:
                                ptr = ptr.right
                            else:
                                ptr = ptr.left
                    else:
                        new_data[i][l] = int(ptr.value)
                        break
            l += 1
    
    # Convert data back into RGB
    final_data = np.empty([900, 900, 3])
    for i in range(900):
        for j in range(900):
            final_data[i][j] = colors[new_data[i][j]]
    final_data = final_data*10


    # create image from numpy array
    PIL_image = Image.fromarray(np.uint8(final_data)).convert('RGB')
    py.imshow(PIL_image)
    py.show()

def tree(x, y, f, height, prevErr):    
    # BASE CASE
    for i in x:
        if i.size == 1 or y.size == 1:
            return Node(float('%.3f'%(np.average(y))))
    
    # FIND VARIABLE
    maxCorr = 0
    for i in range(int(x.size/x[0].size)):
        if np.var(x[i]) == 0.0:
            cor = 0
            continue
        cor = np.abs(corr(x[i], y))
        if cor > maxCorr:
            maxCorr = cor
            var = i

    if maxCorr == 0:
        return Node(float('%.3f'%(np.average(y))))
            
    # FIND SPLIT THRESHOLD
    # sort arrays
    indexs = x[var].argsort()
    sortedY = y[indexs]
    sortedXi = np.sort(x[var])
    
    # check for alpha
    minErr = sys.maxsize
    for i in range(1, sortedXi.size):
        alpha = (sortedXi[i] + sortedXi[i-1]) / 2
        
        leftXi = sortedXi[sortedXi < alpha]
        rightXi = sortedXi[~(sortedXi < alpha)]
        leftY = y[sortedXi < alpha]
        rightY = y[~(sortedXi < alpha)]
       
        if leftXi.size == 0 or rightXi.size == 0:
            continue
        
        err = error(sortedXi, leftXi, leftY, rightXi, rightY, alpha)
        if err < minErr:
            minErr = err
            threshold = alpha
            leftSize = leftXi.size
    
    if minErr == sys.maxsize:
        return Node(float('%.3f'%(np.average(y))))
        
    # SPLIT DATA
    sortedX = np.empty([int(x.size/x[0].size),x[0].size])
    leftX = np.empty([int(x.size/x[0].size),leftSize])
    rightX = np.empty([int(x.size/x[0].size),x[0].size-leftSize])
    for i in range(int(x.size/x[0].size)):
        sortedX[i] = np.sort(x[i])
        leftX[i] = sortedX[i][sortedXi < threshold]
        rightX[i] = sortedX[i][~(sortedXi < threshold)]
        leftY = y[sortedXi < threshold]
        rightY = y[~(sortedXi < threshold)]
            
    # MAKE TREE
    converted = str(var + 1) + " " + str(threshold)
    root = Node(converted)
    
    # CALCULATE TRAINING ERROR
    currErr = minErr
    
    if f >= height:
        return Node(float('%.3f'%(np.average(y))))
    
    # RECURSE
    root.left = tree(leftX, leftY, f+1, height, currErr)
    root.right = tree(rightX, rightY, f+1, height, currErr)
    return root

# Correlation
def corr(x, y):
    covariance = (1/10000)*np.sum(x*y)-((1/10000)*np.sum(x))*((1/10000)*np.sum(y))
    varx = np.var(x)
    vary = np.var(y)
    return covariance / np.sqrt(varx * vary)

# Error
def error(x, leftX, leftY, rightX, rightY, alpha):
    leftErr = (1/leftX.size) * np.sum((leftY - np.average(leftY))**2)
    rightErr = (1/rightX.size) * np.sum((rightY - np.average(rightY))**2)
    return (leftX.size/x.size)*(leftErr) + (rightX.size/x.size)*(rightErr)


if __name__ == "__main__":   
    main()