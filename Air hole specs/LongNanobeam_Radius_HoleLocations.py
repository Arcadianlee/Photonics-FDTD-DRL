""" this code establishes the real-time port b/t FDTD and Python to enable state and reward passes
    for optimization of photonic crystals by RL. NOEL, Renjie Li, December 2021
    The simulation file used here is taken directly from Loncar's paper: Deterministic design of wavelength scale, 
    ultra-high Q photonic crystal nanobeam cavities, whose Q factor = 55E+6. Our goal is to exceed it. 
"""



    def addgeometry(self, l3):
        """ This function constructs the nanobeam geometry by running a setup script in FDTD
        """
        l3.switchtolayout()
        l3.unselectall()
        l3.addstructuregroup()
        l3.set("name", "nanobeam")
        l3.set('x', 0)
        l3.set('y', 0)
        l3.set('z', 0)
        l3.adduserprop("z span", 2, 220E-9)
        l3.adduserprop("N Taper", 0, 15)
        l3.adduserprop("N Mirror", 0, 10)
        l3.adduserprop("index", 0, 1)
        l3.adduserprop("a", 2, 330E-9)
        l3.adduserprop("radius", 0, 0.37)
        #l3.adduserprop("Mirror Radius", 0, 0.3133)
        #l3.adduserprop("ellipse", 0, 1.5)
        l3.adduserprop("k_a", 2, 2.57143E-9)
        l3.adduserprop("Lcavity", 0, 1)
        l3.adduserprop("material", 5, "etch")
        l3.set("construction group", 0)
        a = 330e-9
        Lcavity = 1
        Lcav = Lcavity*a
        radius = 0.37
        z_span = 220e-9
        material = "etch"
        index = 1
        N_Tap = 15
        N_Mir = 10
        k_a = 2.57143E-9

        # ---------------------Taper region (Right)---------------------------
        for i in range(0, N_Tap):
            l3.addcircle()
            #l3.set("make ellipsoid", 1)
            l3.addtogroup("::model::nanobeam")
            l3.set("index", index)
            l3.set("radius", (a*radius-i*k_a))     # semi-minor axis
            l3.set("x", Lcav/2+i*a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        # --------------------Taper region (Left)----------------------
        for i in range(0, N_Tap):
            l3.addcircle()
            #l3.set("make ellipsoid", 1)
            l3.addtogroup("::model::nanobeam")
            l3.set("index", index)
            l3.set("radius", (a*radius-i*k_a))     # semi-minor axis
            l3.set("x", -Lcav/2-i*a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        # -------------------Mirror Region (Right)----------------------
        for i in range(0, N_Mir):
            l3.addcircle()
            #l3.set("make ellipsoid", 1)
            l3.addtogroup("::model::nanobeam")
            l3.set("index", index)
            l3.set("radius", (a*radius-N_Tap*k_a)) # semi-minor axis
            l3.set("x", Lcav/2+N_Tap*a+i*a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        # -------------------Mirror Region (Left)-----------------------
        for i in range(0, N_Mir):
            l3.addcircle()
            #l3.set("make ellipsoid", 1)
            l3.addtogroup("::model::nanobeam")
            l3.set("index", index)
            l3.set("radius", (a*radius-N_Tap*k_a)) # semi-minor axis
            l3.set("x", -Lcav/2-N_Tap*a-i*a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        l3.selectall()
        l3.set("z", 0)
        # if l3.get("material") == "<Object defined dielectric>":
            # l3.set("index", index)



