import numpy as np
import os
# p = os.path.abspath("chio_chih/f1f2_Windt.dat")
# print(p)


def chi0h(energy, element):

    energy = energy*1e3
    element = "  " + element

    # Find a line of the first energy larger than input 'energy'
    with open(os.path.abspath("chi0_chih/f1f2_Windt.dat")) as rf:

        for i, line in enumerate(rf, 1):
            if element in line:
                el_l = i
                break

        for i, line in enumerate(rf, el_l+1):
            if not line.startswith("#"):
                en_l = i
                break

        for i, line in enumerate(rf, en_l+1):
            line = line.strip()
            rb = line.find(" ")
            en = float(line[:rb])
            if en >= energy:
                up_en = i
                break
            if int(en) == 100_000:
                print("End of list")
                break

        p_flag = False
        if en == energy:
            p_flag = True  # if energy in line is the same as in input

    # Calculate linear interpolation of f1, f2
    with open(os.path.abspath("chi0_chih/f1f2_Windt.dat"), "r") as rf:

        # if energy in line is the same as in input
        if p_flag:
            for line in range(up_en-1):
                next(rf)
            line = rf.readline().strip()
            rb = line.find(" ")
            line = line[rb:].strip()
            rb = line.find(" ")
            f1 = float(line[:rb])
            line = line[rb:].strip()
            f2 = float(line)

        # if input energy is between two lines
        else:
            for line in range(up_en-2):
                next(rf)

            # lower energy
            line = rf.readline().strip()
            rb = line.find(" ")
            en_l = float(line[:rb])
            line = line[rb:].strip()
            rb = line.find(" ")
            f1_l = float(line[:rb])
            line = line[rb:].strip()
            f2_l = float(line)

            # upper energy
            line = rf.readline().strip()
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

    # Calculate am, ro
    with open(os.path.abspath("chi0_chih/AtomicConstants.dat"), "r") as rf:

        for line in rf:
            if element in line:
                break

        for line in rf:
            if not line.startswith("#"):
                break

        for i in range(2):
            line = line.strip()
            rb = line.find(" ")
            line = line[rb:]
            line = line.strip()

        rb = line.find(" ")
        am = float(line[:rb])

        for i in range(3):
            line = line.strip()
            rb = line.find(" ")
            line = line[rb:]
            line = line.strip()

        rb = line.find(" ")
        ro = float(line[:rb])

    # Calculate co
    with open(os.path.abspath("chi0_chih/CrossSec-Compton_McMaster.dat"), "r") as rf:

        for line in rf:
            if element in line:
                break

        for line in rf:
            if not line.startswith("#"):
                break

        co = []
        for i in range(3):
            line = line.strip()
            rb = line.find(" ")
            co.append(float(line[:rb]))
            line = line[rb:]
            line = line.strip()

        co.append(float(line))

    # Calculate delta, beta
    e = energy*1e-3
    df2 = (1.4312e-5*e*np.exp(co[0]+co[1]*np.log(e) +
           co[2]*np.log(e)**2+co[3]*np.log(e)**3))
    chi0 = -8.3036e-4*ro/am/e**2*(f1-1j*(f2+df2))
    delta = abs(chi0.real/2)
    beta = abs(chi0.imag/2)

    return delta, beta

delta, beta = chi0h(20.0, 'Si')

print(delta, beta)
