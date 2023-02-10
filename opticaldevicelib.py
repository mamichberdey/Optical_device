from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from chi import chi0h
from concurrent.futures import ThreadPoolExecutor
import os
from collections.abc import Iterable 

if os.path.basename(os.getcwd()) != "CRL_OOP":
    os.chdir("CRL_OOP")

class Optical_device(ABC):
    
    c = 3e8 # speed of light
    dx = 5*10**-9 # step of real-space (лучший шаг для точности/время) # 10k iter for 5*10**-9
    # dx=3*10**-8 # step of real-space
    N = 2**15 # number of points on the grid
    dq = 2*np.pi/(dx*N) # step of reciprocal-space
    lb, rb = -(N-1)/2, N/2
    barr = np.arange(lb, rb) # base array
    x, q = barr*dx, barr*dq # real and reciprocal spaces
    
    def I(self):
            return abs(self.E())**2
    
    def set_z(self, z):
        self.z = z
    
    def ft1d(self, array):
        """
        Calculate normalized Fourier transform of the input 'array';
        return 'ft_array' of the same size
        """
        dx=self.dx
        n = len(array)
        i = np.arange(0, n)
        c_i = np.exp(1j*np.pi*(1-1/n)*i)
        c = np.exp(1j*np.pi*(1-1/2/n))
        ft_array = dx*c*c_i*np.fft.fft(c_i*array)

        return ft_array

    def ift1d(self, array):
        """
        Calculate normalized inverse Fourier transform of the input 'array';
        return 'ift_array' of the same size
        """
        dx=self.dx
        n = len(array)
        i = np.arange(0, n)
        c_i = np.exp(-1j*np.pi*(1-1/n)*i)
        c = np.exp(-1j*np.pi*(1-1/2/n))
        ift_array = 1/dx*c*c_i*np.fft.ifft(c_i*array)

        return ift_array
    
    def P(self, x, z, k):
        """ Fresnel propagator """
        # norm=np.exp(1j*lam*z)/(1j*lam*z)
        # norm=1/(1j*lam*z)
        norm=1
        osn=np.exp(1j*k*(x**2)/(2*z))
        return norm*osn

    def fft_P(self, q, z, k):
        """analytical ft of Fresnel propagator """
        # norm=np.exp(1j*lam*z)/np.sqrt(1j*lam*z)
        # norm=np.sqrt(1j*lam*z)
        # norm=1/np.sqrt(1j*lam*z)
        norm=1
        osn=np.exp(-1j*(q**2)*z/(2*k))
        return norm*osn

    def sv(self, arr, z, k):
        """ convolve arr with analytical ft of Fresnel propagator """
        q=self.q
        return self.ift1d(self.ft1d(array=arr)*self.fft_P(q, z, k))

    @abstractmethod
    
    def E(self):
        pass
  
class Point_source(Optical_device):
    
    def __init__(self, z, En=None, lam=None) -> None:
        super().__init__()
        self.z = z
        if lam==None:
            self.lam = 12.3984e-10/En
        elif En==None:
            self.lam = lam
        else:
            print("lam or E not found!")        
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
    
    def E(self):
        return super().P(x=self.x, z=self.z, k=self.k)

class Hole(Optical_device):
    
    def __init__(self, z, arr_start, D, lam) -> None:
        super().__init__()
        self.z = z
        self.arr_start = arr_start
        self.lam = lam
        self.D = D
        self.En = 12.3984e-10/lam
        self.delta, self.betta = chi0h(energy=self.En, molecula=self.molecula, ro=self.density) 
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
    
    def T2(self, x, D):
        """ "hole" transmission function """
        return  abs(x) <= D/2 
    
    def E(self):
        wfs = self.arr_start
        wfs = self.sv(wfs*self.T2(x=self.x, D=self.D), z=self.z, k=self.k)
        return wfs

class CRL(Optical_device):
    
    def __init__(self, z, arr_start, R, A, d, N1, lam, molecula, density, Flen, gap, copy=0) -> None:
        super().__init__()
        self.gap = gap
        self.molecula = molecula
        self.density = density
        self.Flen = Flen
        self.copy = copy
        self.z = z
        self.arr_start = arr_start
        self.R = R
        self.A = A
        self.d = d
        self.N1 = N1
        self.lam = lam
        self.En = 12.3984e-10/self.lam
        self.delta, self.betta = chi0h(energy=self.En, molecula=self.molecula, ro=self.density) 
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
        
    def FWHM(self, x_arr=None, y_arr=None):
        
        x_arr = self.x if not isinstance(x_arr, Iterable) else x_arr
        y_arr = self.I() if not isinstance(y_arr, Iterable) else y_arr
            
        max = np.max(y_arr)
        half = max/2
        i_ans = np.argwhere(np.diff(np.sign(y_arr - half))).flatten()
        
        x0, y0 = x_arr[i_ans[0]], y_arr[i_ans[0]]
        x1, y1 = x_arr[i_ans[1]], y_arr[i_ans[1]]

        xc0, yc0 = x_arr[i_ans[0]+1], y_arr[i_ans[0]+1]
        xc1, yc1 = x_arr[i_ans[1]+1], y_arr[i_ans[1]+1]

        a0 = (y0-yc0)/(x0-xc0)
        b0 = (y0*xc0-yc0*x0)/(xc0-x0)

        a1 = (y1-yc1)/(x1-xc1)
        b1 = (y1*xc1-yc1*x1)/(xc1-x1)

        x_search_0 = (half-b0)/a0
        x_search_1 = (half-b1)/a1
        
        return x_search_1-x_search_0
        
    def T(self, x=None, R=None, A=None, d=None, Flen=None):
        x = self.x if x == None else x  
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        Flen = self.Flen if Flen == None else Flen
        """returns CRL's thickness"""
        y = x**2/(2*R)
        max = A**2/(8*R)
        y = np.minimum(y, max)
        y = y + d/2
        if Flen != 0:
            y[abs(x)>self.Flen/2] = np.zeros(len(y[abs(x)>self.Flen/2]))
        return y
    
    def Trans(self, delta=None, betta=None, step=0, k=None):
        """ CRL-lense transmission function """
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta
        k = self.k if k == None else k
        # print("int_ideal=", np.sum(T(x, R, A, d)))
        return np.exp(-1j*k*(delta-1j*betta)*(2*self.T()))
    
    def num_wave(self, arr, N1=None, d=None, A=None, R=None, copy=None, k=None):
        
        N1 = self.N1 if N1 == None else N1
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        copy = self.copy if copy == None else copy
        k = self.k if k == None else k

        """ arr_ - E on first CRL-lense's center
            returns E on last lense's center"""
        if N1==0:
            return arr
        p = d+A**2/(4*R)
        w_f = arr

        if N1 == 1:
            t1 = self.Trans(step=0)
            return self.sv(arr=w_f*t1, z=p/2, k=k)

        if copy == True and N1>1:
            t1 = self.Trans(step=0)
            for step in tqdm(range(N1-1)):
                w_f = self.sv(arr=w_f*t1, z=p+self.gap, k=k)
            return self.sv(arr=w_f*t1, z=p/2, k=k)

        elif copy == False and N1>1:
            for step in tqdm(range(N1-1)):
                t1 = self.Trans(step=step)
                w_f = self.sv(arr=w_f*t1, z=p+self.gap, k=k)
            return self.sv(arr=w_f*t1, z=p/2, k=k)
                 
    def E(self):
        wfs = self.arr_start
        p = self.d+self.A**2/(4*self.R)
        wfs = self.sv(arr=wfs, z=p/2, k=self.k)
        wfs = self.sv(arr=self.num_wave(arr=wfs), z=self.z, k=self.k)
        return wfs
    
    def focus(self, N1=None, d=None, A=None, R=None, delta=None, gap=None):
        N1 = self.N1 if N1 == None else N1
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        delta = self.delta if delta == None else delta  
        gap = self.gap if gap == None else gap
        p = d+A**2/(4*R)
        Lc = np.sqrt(p*R/(2*delta))
        x0 = 1
        Th0 = 0
        u = p/Lc
        if N1==1:
            return Lc*np.cos(u)/np.sin(u)
        
        for step in range(N1):
            if step == N1-1:
                gap=0
            x0 = x0*np.cos(u)+Th0*Lc*np.sin(u)-gap*Th0
            Th0 = -x0/Lc*np.sin(u)+Th0*np.cos(u)

        F = -x0/Th0  

        return F

    def image_prop(self, z0, z1, N1=None, delta=None, betta=None, d=None, A=None, R=None, k=None, x=None):
        """ image propagator """

        N1 = self.N1 if N1 == None else N1
        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        k = self.k if k == None else k
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta

        r1 = z1
        p = d+A**2/(4*R)
        eta = delta-1j*(betta)
        zc = np.sqrt(p*R/(2*(eta)))
        L = N1*p
        Cl = np.cos(L/zc)
        Sl = np.sin(L/zc)
        # Ci=Cl-Sl*z1/zc
        C0 = Cl-Sl*z0/zc
        r0 = z0
        rg = (r1+r0)*Cl+(zc-r1*r0/zc)*Sl
        absor_param = np.exp(-1j*k*eta*N1*d)
        # absor_param = 1
        return np.sqrt(r0/rg)*(np.exp(1j*k*(C0*x**2)/(2*rg)))*absor_param
    
    def analytic_solution_CRL(self, z0, z1, N1=None, delta=None, betta=None, d=None, A=None, R=None, k=None, x=None):
        
        N1 = self.N1 if N1 == None else N1
        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        k = self.k if k == None else k
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta

        return abs(self.image_prop(z0=z0, z1=z1, N1=N1, delta=delta, betta=betta, d=d, A=A, R=R, k=k, x=x))**2

class CRLm(CRL):
    
    def __init__(self, b, m, arr_s, arr_w, copy, arr_phase, z, arr_start, R, A, d, N1, lam, molecula, density, Flen, gap) -> None:
        # super().__init__(z, arr_start, R, A, d, N1, lam)
        self.gap = gap
        self.Flen = Flen
        self.molecula = molecula
        self.density = density
        self.z = z
        self.copy = copy
        self.arr_start = arr_start
        self.R = R
        self.A = A
        self.d = d
        self.N1 = N1
        self.lam = lam
        self.En = 12.3984e-10/self.lam
        self.delta, self.betta = chi0h(energy=self.En, molecula=self.molecula, ro=self.density) 
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
        self.b = b
        self.m = m

        if (copy == True and len(arr_w) == 2) or (copy == False and len(arr_w) == self.N1*2):
            self.arr_w = arr_w
        if (copy == True and len(arr_s) == 2) or (copy == False and len(arr_s) == self.N1*2):
            self.arr_s = arr_s
        if (copy == True and len(arr_phase) == 2) or (copy == False and len(arr_phase) == self.N1*2):
            self.arr_phase = arr_phase 
    
    def foo(self, arr, phase, b=None, m=None):
        b = self.b if b == None else b
        m = self.m if m == None else m
        # return b*np.sin(m*x)+b/4*np.cos(m*4*x)+b/8*np.cos(m*8*x)+b/4*np.sin(m*8*x)
        # return b*np.cos(m*x)
        # return b*np.cos(m*arr+rand_phase)
        # phase=2*np.pi*np.random.rand()
        return b*np.cos(m*arr+phase)

    def x_arr(self, arr, phase, s, w, R=None):
        R = self.R if R == None else R
        a = 1/(2*R)
        h = self.A**2/(8*R)
        sq = np.sqrt(1+(2*a*arr)**2)
        xx = arr-2*a*arr*self.foo(arr=arr/2+arr*np.sqrt((a*arr)**2+1/4), phase=phase)/sq
        yy = a*arr**2+self.foo(arr=arr/2+arr*np.sqrt((a*arr)**2+1/4), phase=phase)/sq
        # x = s+xx*np.cos(w)-(yy-h)*np.sin(w)
        x = s+xx*np.cos(w)-(yy)*np.sin(w)
        return x

    def y_arr(self, arr, phase, w, R=None):
        R = self.R if R == None else R
        a = 1/(2*R)
        h = self.A**2/(8*R)
        sq = np.sqrt(1+(2*a*arr)**2)
        xx = arr-2*a*arr*self.foo(arr=arr/2+arr*np.sqrt((a*arr)**2+1/4), phase=phase)/sq
        yy = a*arr**2+self.foo(arr=arr/2+arr*np.sqrt((a*arr)**2+1/4), phase=phase)/sq
        # y = h+xx*np.sin(w)+(yy-h)*np.cos(w)  
        y = xx*np.sin(w)+(yy)*np.cos(w) 
        return y
    
    def T(self, phase, s, w, x=None, R=None, A=None, d=None, b=None, m=None, N=None, Flen=None):
        import time
        """returns lens thickness considering rough surface"""

        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        b = self.b if b == None else b
        m = self.m if m == None else m
        N = self.N if N == None else N
        Flen = self.Flen if Flen == None else Flen
        
        h = A**2/(8*R)
        
        minstep = x[1]-x[0]
        t = x
        dt = minstep

        yyy_for_process = self.y_arr(arr=t, phase=phase, w=w)

        
        low = np.min(yyy_for_process)


        t = t[yyy_for_process<=h] # новое поле t

        
        dt = t[1]-t[0]
        
        xxx_for_process = self.x_arr(arr=t, phase=phase, s=s, w=w)
        # print(len(xxx_for_process))

        x_start, x_end = np.min(xxx_for_process), np.max(xxx_for_process) # создание нового равномерного поля x (равномеризация сетки)
        x_space = np.arange(x_start, x_end, minstep)
        dx = x_space[1]-x_space[0]
        

        


        def process_key_value(x0):
            i_ans = np.argwhere(np.diff(np.sign(x0 - xxx_for_process))).flatten() # поиск пересечения с осью x0
            x_fooi = xxx_for_process[i_ans]
            x_fooiplus = xxx_for_process[i_ans+1]
            t_new = (x0-(x_fooi-((x_fooiplus-x_fooi)/dt)*t[i_ans]))/((x_fooiplus-x_fooi)/dt)  # линейное интерполированное под пересечение значение t
            y_space = self.y_arr(arr=t_new, phase=phase, w=w)
            y_space = np.concatenate(([h], y_space, [low]))
            y_interi = np.sum(-np.diff(y_space)[-1::-2])  
            return y_interi
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            res = executor.map(process_key_value, (x_space))
  
        y_inter = np.array(list(res))

        y_inter = np.minimum(y_inter, h-low)

        # y_inter = []
        # for x0 in x_space:
        #     i_ans = np.argwhere(np.diff(np.sign(x0 - xxx_for_process))).flatten() # поиск пересечения с осью x0
        #     x_fooi = xxx_for_process[i_ans]
        #     x_fooiplus = xxx_for_process[i_ans+1]
        #     t_new = (x0-(x_fooi-((x_fooiplus-x_fooi)/dt)*t[i_ans]))/((x_fooiplus-x_fooi)/dt)  # линейное интерполированное под пересечение значение t
        #     y_space = self.y_arr(arr=t_new, phase=phase, w=w)
        #     y_space = np.concatenate(([h], y_space, [low]))
        #     y_interi = np.sum(-np.diff(y_space)[-1::-2])  
        #     y_inter.append(y_interi)

        # y_inter = np.array(y_inter)    
        # y_inter = np.minimum(y_inter, h-low)    

        time1 = time.perf_counter()    
        zeroind = int(np.where(x_space == x_space[x_space < 0][-1])[0][0])+1
        alph = int(N/2-zeroind)
        bett = int(N/2-len(x_space)+zeroind)
        x_space_ext1 = np.linspace(x_space[0]-dx, x_space[0]-(alph)*dx, alph)
        x_space_ext2 = np.linspace(x_space[-1]+dx, x_space[-1]+(bett)*dx, bett) 
        x_space = np.concatenate((x_space_ext1[::-1], x_space, x_space_ext2))
        
        y_inter_ext1 = np.ones(len(x_space_ext1))*(h-low)
        y_inter_ext2 = np.ones(len(x_space_ext2))*(h-low)
        y_inter = np.concatenate((y_inter_ext1, y_inter, y_inter_ext2))
        
        y_inter = y_inter+low+d/2


        time2 = time.perf_counter()
        print("t = ", time2- time1)

        if Flen != 0:
            y_inter[abs(x_space)>self.Flen/2] = np.zeros(len(y_inter[abs(x_space)>self.Flen/2]))

        return y_inter   
 
    def wavy_parabola(self, phase, x=None, R=None, A=None, d=None, b=None, m=None, N=None):
        
        """returns lens thickness considering rough surface"""

        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        b = self.b if b == None else b
        m = self.m if m == None else m
        N = self.N if N == None else N
        
        h = A**2/(8*R)
        t = x
        
        yyy_for_process = self.y_arr(arr=t, phase=phase)
        
        low = np.min(yyy_for_process)
        
        t = t[yyy_for_process<=h] # новое поле t

        x_space = self.x_arr(t, phase)
        dx = x_space[1]-x_space[0]
        
        y_inter = self.y_arr(t, phase)
    
                
        # zeroind = int(np.where(x_space == x_space[x_space < 0][-1])[0][0])+1
        # alph = int(N/2-zeroind)
        # bett = int(N/2-len(x_space)+zeroind)
        # x_space_ext1 = np.linspace(x_space[0]-dx, x_space[0]-(alph)*dx, alph)
        # x_space_ext2 = np.linspace(x_space[-1]+dx, x_space[-1]+(bett)*dx, bett) 
        # x_space = np.concatenate((x_space_ext1[::-1], x_space, x_space_ext2))
        
        # y_inter_ext1 = np.ones(len(x_space_ext1))*(h-low)
        # y_inter_ext2 = np.ones(len(x_space_ext2))*(h-low)
        # y_inter = np.concatenate((y_inter_ext1, y_inter, y_inter_ext2))
        
        y_inter = y_inter+d/2
        # print(y_inter)

        return x_space, y_inter   
  
    def Trans(self, step=None, delta=None, betta=None, k=None):
        """ CRL-lense transmission function """
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta
        k = self.k if k == None else k
        
        if (len(self.arr_phase) == 2 or len(self.arr_phase) == self.N1*2) \
            and (len(self.arr_w) == 2 or len(self.arr_w) == self.N1*2)\
                and (len(self.arr_s) == 2 or len(self.arr_s) == self.N1*2):
            T_arr1 = self.T(phase=self.arr_phase[0+step], w=self.arr_w[0+step], s=self.arr_s[0+step]) 
            T_arr2 = self.T(phase=self.arr_phase[1+step], w=self.arr_w[1+step], s=self.arr_s[1+step]) 
        else:
            print("Wrong number of phases!")    

        return np.exp(-1j*k*(delta-1j*betta)*(T_arr1+T_arr2)) 
             
class CRL3(CRL):
    
    def __init__(self, z, arr_start, R, A, d, N1, lam) -> None:
        super().__init__(z, arr_start, R, A, d, N1, lam)
        self.z = z
        self.arr_start = arr_start
        self.R = R
        self.A = A
        self.d = d
        self.N1 = N1
        self.lam = lam
        self.En = 12.3984e-10/lam
        self.delta, self.betta = chi0h(energy=self.En, molecula=self.molecula, ro=self.density) 
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
    
    # def T(self, x, R, A, d):
    #     """ CRL-lense transmission function (cubic, A=const) """
    #     eps_T3=(1/R)*0.01
    #     max=(A**2/(8*R))
    #     T_for_T3=(1/(2*R)+eps_T3)*x**2-eps_T3*abs(x)**3/np.sqrt(2*max*R)
    #     T_for_T3[abs(x)>=np.sqrt(2*max*R)]=max
    #     T_for_T3=T_for_T3+d/2
    #     return T_for_T3

    def T(self, x, R, A, d):
        """ CRL-lense transmission function (cubic, A!=const, A_new=A_old+2*eps) """
        max = (A**2/(8*R))
        eps = A*1/100
        skobka = (np.sqrt(2*max*R)+eps)
        T_for_T3 = (1/(2*R))*x**2+(max/skobka**3-1/(2*R*skobka))*abs(x)**3
        T_for_T3[abs(x) >= np.sqrt(2*max*R)+eps]=max
        T_for_T3 = T_for_T3+d/2
        return T_for_T3 

if __name__ == "__main__":
    import time

    t1 = time.perf_counter()
    
    """Добавление библиотеки в код"""
    import opticaldevicelib as od

    """Способ инициализации точечного источника с энергией En и распр. ВФ на расстоянии z от него"""
    p = od.Point_source(z=100, En=10) 
    E_arr = p.E()

    """Пример инициализации пластиковых (C21SO8SiH36) линз (идеальная и неидеальная)"""

    # N1_global = 10
    # Copy_flag = False
    # arr_len = 2 if Copy_flag else 2*N1_global

    # phases = 2*np.pi*np.random.rand(arr_len)
    # w_s = (np.random.rand(arr_len)-0.5)*np.pi/180*1
    # s_s  = (np.random.rand(arr_len)-0.5)*2e-6*1

    # crl = od.CRL(lam=p.lam, arr_start=E_arr,\
    #                 R=5e-6, A=24e-6, d=5e-6, N1=N1_global, z=0,\
    #                     molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0)

    # crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
    #                     R=5e-6, A=24e-6, d=5e-6, N1=N1_global, z=0,\
    #                         molecula="C21SO8SiH36", density=1.12, Flen=0, gap=0,\
    #                             b=1e-6, m=3e6, copy=Copy_flag, arr_phase=phases, arr_s=s_s, arr_w=w_s)
                                
    """Пример инициализации кремниевых (Si) линз (идеальная и неидеальная)"""

    N1_global = 10
    Copy_flag = False
    arr_len = 2 if Copy_flag else 2*N1_global

    phases = 2*np.pi*np.random.rand(arr_len)
    w_s = (np.random.rand(arr_len)-0.5)*np.pi/180*1
    s_s  = (np.random.rand(arr_len)-0.5)*2e-6*1

    crl = od.CRL(lam=p.lam, arr_start=E_arr,\
                    R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                        molecula="Si", density=2.33, Flen=0, gap=0)

    crlm = od.CRLm(lam=p.lam, arr_start=E_arr,\
                        R=6.25e-6, A=50e-6, d=2e-6, N1=N1_global, z=0,\
                            molecula="Si", density=2.33, Flen=0, gap=0,\
                                b=1e-6, m=3e6, copy=Copy_flag, arr_phase=phases, arr_s=s_s, arr_w=w_s)
    
    """ Распределение интенсивности излучения в фокусе """

    """ !!!Важно!!! Поиск фокусного расстояния и установка"""

    focus = crl.focus() 
    crl.set_z(z=focus)
    crlm.set_z(z=focus) 

    import matplotlib.pyplot as plt

    plt.plot(p.x, crl.I(), label="numerical CRL")
    plt.plot(p.x, crlm.I(), label="numerical CRLm")
    t2 = time.perf_counter()
    print(t2-t1)
    plt.legend()
    plt.grid()  
    plt.show()                            
