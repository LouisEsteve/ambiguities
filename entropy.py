from math import log2
def entropy(x) -> float:
    d = {}
    x = [str(i) for i in x]
    # avg. O(n)
    for i in x:
        if i not in d.keys():
            d[i] = 1
        else:
            d[i] += 1
    sum = 0.0
    len_x = len(x)
    for i in d.keys():
        p = d[i] / len_x
        sum += p * log2(p)
    # print(d)
    return -sum



# print(entropy([0,0,0,0]))
# print(entropy([0,0,1,0]))
# print(entropy([0,0,2,0]))
# print(entropy([0,0,1,1]))