""" this code establishes the real-time port b/t FDTD and Python to enable state and reward passes
    for optimization of photonic crystals by RL. NOEL, Renjie Li, December 2021
"""



class FdtdRlNanobeam():

    def __init__(self):

        self.numMirror = 9
        self.numTaper = 4
        #self.cavList = [1, 3, 5]  # different cavity numbers to choose from

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
        l3.adduserprop("z_span", 2, 500E-9)
        l3.adduserprop("shift_hole", 0, 0.03)
        l3.adduserprop("N", 0, 8)
        l3.adduserprop("radius", 0, 0.24)
        l3.adduserprop("index", 0, 1)
        l3.adduserprop("a", 2, 250e-9)
        l3.adduserprop("delta_R2", 0, 0.03)
        l3.adduserprop("delta_a", 0, 0.06)
        l3.adduserprop("ellipse", 0, 1.5)
        l3.adduserprop("Lcavity", 0, 1.18)
        l3.adduserprop("material", 5, "etch")
        l3.set("construction group", 0)
        L_number = 3
        a = 250e-9
        delta_a = 0.06
        delta_R2 = 0.03
        shift_hole = 0.03
        Lcavity = 1.18
        Lcav = Lcavity * a
        da = a * delta_a
        deltaR2 = delta_R2 * a
        SHIFT = shift_hole * a
        a4 = a - da
        a3 = a4 - da
        a2 = a3 - da
        a1 = a2 - da
        radius = 0.24
        ellipse = 1.5
        z_span = 500e-9
        material = "etch"
        index = 1
        N = 8

        # ---------------------out of taped--ellipse region---------------------------
        for i in range(0, N+1):
            l3.addcircle()
            l3.set("make ellipsoid", 1)
            l3.set("index", index)
            l3.addtogroup("::model::nanobeam")
            l3.set("radius", (a * radius))     # semi-minor axis
            l3.set("radius 2", (ellipse * a * radius))   # semi-major axis
            l3.set("x", Lcav / 2 + a1 + a2 + a3 + a4 + i * a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        # -------------------out of taped-----ellipse region----------------------
        for i in range(0, N+1):
            l3.addcircle()
            l3.set("make ellipsoid", 1)
            l3.set("index", index)
            l3.addtogroup("::model::nanobeam")
            l3.set("radius", (a * radius))
            l3.set("radius 2", (ellipse * a * radius))
            l3.set("x", -(Lcav/2+a1+a2+a3+a4)-i*a)
            l3.set("y", 0)
            l3.set("z span", z_span)
            l3.set("material", material)

        # -------------------taped region----ellipse3 region----------------------
        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+1*deltaR2))
        l3.set("x", Lcav/2+SHIFT)
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+2*deltaR2))
        l3.set("x", Lcav/2+a1)
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        # -------------------taped region----ellipse2 region--------------------
        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+3*deltaR2))
        l3.set("x", Lcav/2+a1+a2)
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+4*deltaR2))
        l3.set("x", Lcav/2+a1+a2+a3)
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        # -------------------taped region--------------------------
        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+1*deltaR2))
        l3.set("x", -Lcav/2-SHIFT)
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+2*deltaR2))
        l3.set("x", -(Lcav/2+a1))
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        # -------------------taped region----ellipse2 region--------------------

        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+3*deltaR2))
        l3.set("x", -(Lcav/2+a1+a2))
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        l3.addcircle()
        l3.set("make ellipsoid", 1)
        l3.set("index", index)
        l3.addtogroup("::model::nanobeam")
        l3.set("radius", (a * radius))
        l3.set("radius 2", (a*radius+4*deltaR2))
        l3.set("x", -(Lcav/2+a1+a2+a3))
        l3.set("y", 0)
        l3.set("z span", z_span)
        l3.set("material", material)

        l3.selectall()
        l3.set("z", 0)
        # if l3.get("material") == "<Object defined dielectric>":
            # l3.set("index", index)

