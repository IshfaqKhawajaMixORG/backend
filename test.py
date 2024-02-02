# Code PAscals Triangle

def pascals_triangle(x):
    trow = [1]
    y = [0]
    res = trow.copy()
    for n in range(max(x + 1,0)):
        # print(trow)
        res = trow.copy()
        trow=[l+r for l,r in zip(trow+y, y+trow)]
    return res

print(pascals_triangle(4))