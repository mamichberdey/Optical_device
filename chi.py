import numpy as np
import os

def chi0h(energy, molecula, ro):   #molecula in format like 'AlSe8O15'
    F1 = []
    F2 = []
    Co = []
    Df2 = []
    Am = []
    energy = energy*1e3

    MOL = disassemble(molecula)

    for element in MOL['elements']: #Для каждого атома нужно найти значения

        with open(os.path.abspath("chi0_chih/f1f2_Windt.dat") , "r") as f:

            rf = f.readlines()
            l = len(rf)
            for i in range(l):
                if rf[i].startswith("#S"):
                    rf[i] = rf[i].strip()
                    if element in rf[i] and element[-1] == rf[i][-1]: # ДОБАВИЛ ДОПОЛНИТЕЛЬНОЕ УСЛОВИЕ
                        el_l = i
                        # print(rf[i])  # ПРОВЕРКА СООТВЕТСТВИЯ ЭЛЕМЕНТОВ
                        # print(element)  # -//-
                        break

            for i in range(el_l+1,l):
                if not rf[i].startswith("#"):
                    en_l = i
                    break

            for i in range(en_l+1,l):
                line = rf[i].strip()
                rb = line.find(" ")
                en = float(line[:rb])
                if en >= energy:
                    up_en = i
                    break
                if int(en) == 100000:
                    print("End of list")
                    break


            p_flag = False
            if en == energy:
                p_flag = True

            if p_flag:
                line = rf[up_en].strip()
                rb = line.find(" ")
                line = line[rb:].strip()
                rb = line.find(" ")
                f1 = float(line[:rb])
                line = line[rb:].strip()
                f2 = float(line)

            else:
                # lower energy
                line = rf[up_en-1].strip()
                rb = line.find(" ")
                en_l = float(line[:rb])
                line = line[rb:].strip()
                rb = line.find(" ")
                f1_l = float(line[:rb])
                line = line[rb:].strip()
                f2_l = float(line)

                # upper energy
                line = rf[up_en].strip()
                en_u = float(line[:rb])
                rb = line.find(" ")
                line = line[rb:].strip()
                rb = line.find(" ")
                f1_u = float(line[:rb])
                line = line[rb:].strip()
                f2_u = float(line)

                # interpolation f1
                a1 = (f1_u-f1_l)/(en_u-en_l)
                b1 = f1_u - a1*en_u
                f1 = energy*a1+b1

                # interpolation f2
                a2 = (f2_u-f2_l)/(en_u-en_l)
                b2 = f2_u - a2*en_u
                f2 = energy*a2+b2

            F1.append(f1)
            F2.append(f2)

        with open(os.path.abspath("chi0_chih/CrossSec-Compton_McMaster.dat"), "r") as f:

            rf = f.readlines()
            l = len(rf)

            for i in range (l):
                if rf[i].startswith("#S"):
                    rf[i] = rf[i].strip()
                    if element in rf[i] and element[-1] == rf[i][-1]:  # ДОБАВИЛ ДОПОЛНИТЕЛЬНОЕ УСЛОВИЕ
                        c=i
                        # print(rf[i])  # ПРОВЕРКА СООТВЕТСТВИЯ ЭЛЕМЕНТОВ
                        # print(element)  # -//-
                        break

            for i in range (c+1,l):
                if not rf[i].startswith("#"):
                    c=i
                    break

            co = []
            line = rf[c]
            for i in range(3):
                line = line.strip()
                rb = line.find(" ")
                co.append(float(line[:rb]))
                line = line[rb:]
                line = line.strip()

            co.append(float(line))
            Co.append(co)

        with open(os.path.abspath("chi0_chih/AtomicConstants.dat"), "r") as f:

            rf = f.readlines()
            l = len(rf)

            for i in range (l):
                if rf[i].startswith("#S"):
                    rf[i] = rf[i].strip()
                    if element in rf[i] and element[-1] == rf[i][-1]:
                        c=i
                        break

            for i in range (c+1,l):
                if not rf[i].startswith("#"):
                    c=i
                    break

            line=rf[c]
            for i in range(2):
                line = line.strip()
                rb = line.find(" ")
                line = line[rb:]
                line = line.strip()

            rb = line.find(" ")
            am = float(line[:rb])

            Am.append(am)
            
    e = energy*1e-3

    for i in range(len(Co)):  #Cчитает df2 для каждого атома
        co = Co[i]
        df2 = (1.4312e-5*e*np.exp(co[0]+co[1]*np.log(e)+co[2]*np.log(e)**2+co[3]*np.log(e)**3))
        Df2.append(df2)


    Am = np.array(Am)
    Mol = np.sum(Am*MOL['indexes']) 

    smth=atomic_sum(MOL,F1,F2,Df2) 

    chi0 = -8.3036e-4*ro/Mol/e**2*smth
    delta = abs(chi0.real/2)
    beta = abs(chi0.imag/2)

    return [delta,beta]

def atomic_sum(MOL,F1,F2,Df2):
    smth=0
    for i in range(len(MOL['indexes'])):
        f1=F1[i]
        f2=F2[i]
        df2=Df2[i]
        smth=smth+(f1-1j*(f2+df2))*MOL['indexes'][i]

    return smth

def disassemble(molecula):
    molecula = molecula+' '
    elements = []
    indexes = []
    point = 0
    flag = 1
    i = 1
    eflag = 0
    while i<(len(molecula)):
        if not molecula[i].islower() and flag==1:
            elements.append("  " +molecula[point:i])
            point = i
            count = i
            flag=0
            while count<len(molecula) and (molecula[count].isdigit() or molecula[count]=='.' or molecula[count]==','):
                count += 1
            if count == i:
                indexes.append(1)
            else:
                indexes.append(float(molecula[i:count]))
            point = count
            i=count
        else:
            flag = 1
            i += 1
        # print(elements)


    return {'elements':elements,'indexes':indexes}

if __name__ == "__main__":
    delta, beta = chi0h(10, 'C21SO8SiH36', 1.12)
    print(delta)