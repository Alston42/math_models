'''
@params:
T0: initial temperature, a big number
d: cooling coefficient, a number close to 1 but less than 1
Tk: termination temperature, a positive number close to 0
'''

import random
import math

N = 10005
n = 0
x = [0] * N
y = [0] * N
w = [0] * N
ansx, ansy, dis = 0, 0, 0

def calc(xx, yy):
    res = 0
    for i in range(1, n + 1):
        dx = x[i] - xx
        dy = y[i] - yy
        res += math.sqrt(dx * dx + dy * dy) * w[i]
    global dis
    if res < dis:
        dis = res
        ansx = xx
        ansy = yy
    return res

def Rand():
    return random.random()

def simulate_annealing():
    t = 100000
    nowx, nowy = ansx, ansy
    while t > 0.001:
        nxtx = nowx + t * (Rand() * 2 - 1)
        nxty = nowy + t * (Rand() * 2 - 1)
        delta = calc(nxtx, nxty) - calc(nowx, nowy)
        if math.exp(-delta / t) > Rand():
            nowx, nowy = nxtx, nxty
        t *= 0.97

    for i in range(1, 1001):
        nxtx = ansx + t * (Rand() * 2 - 1)
        nxty = ansy + t * (Rand() * 2 - 1)
        calc(nxtx, nxty)

if __name__ == "__main__":
    n = int(input())
    for i in range(1, n + 1):
        x[i], y[i], w[i] = map(int, input().split())
        ansx += x[i]
        ansy += y[i]
    ansx /= n
    ansy /= n
    calc(ansx, ansy)
    simulate_annealing()
    print("%.3lf %.3lf" % (ansx, ansy))