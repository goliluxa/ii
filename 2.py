from threading import *

l = list()

def go():
    global l
    for i in range(1000000000):
        a = i**i
        if i % 2 == 0:
            l.append(a)
        else:
            l.append(a + 1)


def starting():
    for i in range(10):
        Thread(target=go).start()

def startin2():
    for i in range(50):
        Thread(target=starting).start()
        print(active_count())

Thread(target=startin2()).start()
