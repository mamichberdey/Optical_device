
class A:
    
    def __init__(self, N) -> None:
        self.N = N
        

class B(A):
    
    def __init__(self, k) -> None:
        self.k = k 
    

p = A(N=3)
B.N = 3
b = B(k=5)
print(b.N)