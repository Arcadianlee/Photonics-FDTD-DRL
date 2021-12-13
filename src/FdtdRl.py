""" this code establishes the real-time port b/t FDTD and Python to enable state and reward passes
    for optimization of photonic crystals by RL. NOEL, Renjie Li, June 2021
"""

import numpy as np
import random
import gym
import sys
import os
sys.path.append("C:\\Program Files\\Lumerical\\v202\\api\\python\\")   # Default windows lumapi path
sys.path.append(os.path.dirname(__file__))   # Current directory
import lumapi as lp

class FdtdRl():

    def __init__(self):

        pass

    def addgeometry(self, l3):
        """ This function constructs the L3 geometry by running a setup script in FDTD
        """
        l3.switchtolayout()
        l3.unselectall()
        l3.addstructuregroup()
        l3.set("name", "L3_PC1")
        l3.set('x', 0)
        l3.set('y', 0)
        l3.set('z', 0)
        l3.adduserprop("z_span", 2, 340E-9)
        l3.adduserprop("L_number", 0, 3)
        l3.adduserprop("dx_shift", 0, 0.15)
        l3.adduserprop("dr_shrink", 0, 0)
        l3.adduserprop("n_x", 0, 8)
        l3.adduserprop("n_y", 0, 8)
        l3.adduserprop("index", 0, 1)
        l3.adduserprop("a", 2, 320E-9)
        l3.adduserprop("radius", 2, 89.6E-9)
        l3.adduserprop("material", 5, "etch")
        l3.set("construction group", 0)
        L_number = 3
        n_x = 8
        n_y = 8
        a = 320e-9
        dx_shift = 0.15
        dr_shrink = 0
        xs = l3.mod(L_number, 2)/2
        xs0 = (1 / 2 - xs)
        radius = 89.6e-9
        z_span = 340e-9
        material = "etch"
        index = 1

        # draw center row
        for j in range(-(n_x + 0), (n_x + 0)+1):
            if abs(j) >= L_number/2 + 1:

                l3.addcircle()
                l3.set("index", index)
                l3.addtogroup("::model::L3_PC1")

                l3.set("x", (j) * a)
                l3.set("y", 0)
                l3.set("radius", radius)
            else:
                if abs(j) >= L_number/2:
                    l3.addcircle()
                    l3.set("index", index)
                    l3.addtogroup("::model::L3_PC1")

                    l3.set("x", (j) * (a + dx_shift * a / abs(j)))
                    l3.set("y", 0)
                    l3.set("radius", radius - (dr_shrink * a))

        # draw upper and lower rows
        for i in range(1, n_y, 2):
            for j in np.arange(-(n_x+xs), (n_x + xs)+1):
                l3.addcircle()
                l3.set("index", index)
                l3.addtogroup("::model::L3_PC1")

                l3.set("x", (j) * a)
                l3.set("y", i * a * l3.sqrt(3) / 2)
                l3.set("radius", radius)

                l3.addcircle()
                l3.set("index", index)
                l3.addtogroup("::model::L3_PC1")

                l3.set("x", (j) * a)
                l3.set("y", -i * a * l3.sqrt(3) / 2)
                l3.set("radius", radius)

            i = i + 1

            for j in range(-(n_x+0), (n_x + 0)+1):
                l3.addcircle()
                l3.set("index", index)
                l3.addtogroup("::model::L3_PC1")

                l3.set("x", (j) * a)
                l3.set("y", i * a * l3.sqrt(3) / 2)
                l3.set("radius", radius)

                l3.addcircle()
                l3.set("index", index)
                l3.addtogroup("::model::L3_PC1")

                l3.set("x", (j) * a)
                l3.set("y", -i * a * l3.sqrt(3) / 2)
                l3.set("radius", radius)


        l3.selectall()
        l3.set("z", 0)
        l3.set("z span", z_span)
        #l3.set("material", material)

        l3.runsetup()


    def adjustdesignparams(self, dx1, dy1, dr1):
        """ This function makes is convenient to reconstruct the simulation;
                while changing the design parameters, a brand new FDTD session will start
                and close within this function. Symmetry of the geometry is taken into account.
        """

        print("starting new FDTD session... ")

        with lp.FDTD() as l3:
            l3.load("L3-PhC-RL.fsp")

            # create the PC structure setup script
            self.addgeometry(l3)

            dr = dr1  # add random radius changes
            # dx1, dy1, dr1 default to be the first quadrant

            for i in range(4, 11+1):   # center row
                j = i - 4   # j = 0:8
                if i < 8:  # left hand side
                    rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                    l3.setnamed("::model::L3_PC1::circle", "radius", float(rad+dr), i)

                    dx = -dx1  # add random x, y displacements
                    dy = 0
                    px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                    l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                    py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                    l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                else:  # right hand side
                    rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                    l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                    dx = dx1  # add random x, y displacements
                    dy = 0
                    px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                    l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                    py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                    l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)

            for i in range(21, 44+1):   # second row
                j = i - 13  # 8:32
                if i % 2:   # odd number (top row)
                    if i < 32:  # 2nd quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = -dx1  # add random x, y displacements
                        dy = dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    else:  # 1st quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = dx1  # add random x, y displacements
                        dy = dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)

                else:  # even number (bottom row)
                    if i < 33:  # 3rd quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad+dr), i)
                        dx = -dx1  # add random x, y displacements
                        dy = -dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    else:  # 4th quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = dx1  # add random x, y displacements
                        dy = -dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)

            for i in range(57, 78+1):   # third row
                j = i - 25  # 32:54
                if i % 2:   # odd number (top row)
                    if i < 66:  # 2nd quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = -dx1  # add random x, y displacements
                        dy = dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    elif i == 67:  # center vertical column
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = 0  # add random x, y displacements
                        dy = dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    else:  # 1st quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = dx1  # add random x, y displacements
                        dy = dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)

                else:  # even number (bottom row)
                    if i < 67:  # 3rd quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad+dr), i)
                        dx = -dx1  # add random x, y displacements
                        dy = -dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    elif i == 68:  # center vertical column
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = 0  # add random x, y displacements
                        dy = -dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)
                    else:  # 4th quadrant
                        rad = l3.getnamed("::model::L3_PC1::circle", "radius", i)
                        l3.setnamed("::model::L3_PC1::circle", "radius", float(rad + dr), i)
                        dx = dx1  # add random x, y displacements
                        dy = -dy1
                        px = l3.getnamed("::model::L3_PC1::circle", "x", i)
                        l3.setnamed("::model::L3_PC1::circle", "x", float(px + dx), i)
                        py = l3.getnamed("::model::L3_PC1::circle", "y", i)
                        l3.setnamed("::model::L3_PC1::circle", "y", float(py + dy), i)

            l3.run()

            pxNew = l3.getnamed("::model::L3_PC1::circle", "x", 21)
            pyNew = l3.getnamed("::model::L3_PC1::circle", "y", 21)

            radNew = l3.getnamed("::model::L3_PC1::circle", "radius", 21)
            pxNew1 = l3.getnamed("::model::L3_PC1::circle", "x", 44)
            pyNew1 = l3.getnamed("::model::L3_PC1::circle", "y", 44)

            print(pxNew, pyNew, pxNew1, pyNew1)

            l3.runanalysis()
            Qraw = l3.getresult("::model::Q::Qanalysis", "Q")
            Qmax = max(Qraw['Q'])

            print(Qmax)

            Vraw = l3.getresult("::model::Q::mode_volume", "Volume")

            l3.switchtolayout()
            l3.select("::model::L3_PC1")
            l3.delete()
            l3.save()

        return Qmax, Vraw, pxNew, radNew


    # mn = 0   # mean in nm
    # stddev = 0.5E-9    # standard deviation in nm
    #
    # # modify the design parameters
    # DR = np.ones(54)*random.uniform(0,1)*stddev  # change in radius
    # DX = np.ones(54)*random.uniform(0,2)*stddev  # change in x coordinate
    # DY = np.ones(54)*random.uniform(0,2)*stddev   # change in y coordinate
    #
    # # run the simulation
    # Qf, Vm, x, r = adjustdesignparams(DR, DX, DY)
    #
    # # calculate Q factor and modal volume
    # Q = max(Qf['Q'])
    # V = min(Vm['V'])
    #
    # print(DR[0])
    # print(Q)
    # print(V)
    # print(x)
    # print(r)

