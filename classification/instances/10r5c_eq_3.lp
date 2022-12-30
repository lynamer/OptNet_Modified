minimize
OBJ: +89 x1 +25 x2 +68 x3 +83 x4 +4 x5

subject to
C0: +1 x1 +1 x4 +1 x5 >= 1
C1: +1 x1 +1 x4 +1 x5 >= 1
C2: +1 x1 +1 x3 +1 x4 +1 x5 >= 1
C3: +1 x1 +1 x4 >= 1
C4: +1 x1 +1 x3 +1 x4 +1 x5 >= 1
C5: +1 x1 == 1
C6: +1 x2 == 1
C7: +1 x1 +1 x3 >= 1
C8: +1 x2 +1 x4 >= 1
C9: +1 x1 +1 x3 +1 x5 >= 1

binary
 x1 x2 x3 x4 x5