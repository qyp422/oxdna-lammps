'''
Author: qyp422
Date: 2022-09-22 15:40:48
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-04-02 15:03:33
FilePath: \dump\oxdnaenergy\mathfuction.py
Description: 
Rotation matrix and Quaternion and eulerAngles
f1~f6 base functions of oxdna2 model

Copyright (c) 2022 by qyp422, All Rights Reserved. 
'''
from numba import njit
import scipy.interpolate
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation 
plt.rc('font', family = 'Arial',size = 18)
from scipy.stats import gaussian_kde



import base
print('\n' + __file__ + ' is called\n')

'''
description:Quaternion to Rotation matrix just for standard quat
param {*} myquat w,x,t,z must be nparray or list
return {*} a1,a2,a3
'''


@njit
def q_to_exyz (myquat):
    sqw = myquat[0] * myquat[0]
    sqx = myquat[1] * myquat[1]
    sqy = myquat[2] * myquat[2]
    sqz = myquat[3] * myquat[3]

#    invs = 1 / (sqx + sqy + sqz + sqw) 
    invs = 1 # for standard quat is 1
    m00 = (sqx - sqy - sqz + sqw) * invs 
    m11 = (-sqx + sqy - sqz + sqw) * invs 
    m22 = (-sqx - sqy + sqz + sqw) * invs 
    
    tmp1 = myquat[1] * myquat[2]
    tmp2 = myquat[3] * myquat[0]
    m10 = 2.0 * (tmp1 + tmp2) * invs 
    m01 = 2.0 * (tmp1 - tmp2) * invs 

    tmp1 = myquat[1] * myquat[3]
    tmp2 = myquat[2] * myquat[0]
    m20 = 2.0 * (tmp1 - tmp2) * invs 
    m02 = 2.0 * (tmp1 + tmp2) * invs 
    tmp1 = myquat[2] * myquat[3]
    tmp2 = myquat[1] * myquat[0]
    m21 = 2.0 * (tmp1 + tmp2) * invs 
    m12 = 2.0 * (tmp1 - tmp2) * invs 

    mya1 = np.array([m00, m10, m20])
    mya2 = np.array([m01, m11, m21])
    mya3 = np.array([m02, m12, m22])

    return mya1, mya2 ,mya3


'''
description: Rotation matrix to Quaternion
param {*} mya1 a1
param {*} mya3 a3
return {*} quat w,x,y,z
'''
def exyz_to_quat (mya1, mya3):
    
    mya2 = np.cross(mya3, mya1)
    myquat = np.array([1., 0., 0., 0.])

    q0sq = 0.25 * (mya1[0] + mya2[1] + mya3[2] + 1.0)
    q1sq = q0sq - 0.5 * (mya2[1] + mya3[2])
    q2sq = q0sq - 0.5 * (mya1[0] + mya3[2])
    q3sq = q0sq - 0.5 * (mya1[0] + mya2[1])

    # some component must be greater than 1/4 since they sum to 1
    # compute other components from it

    if q0sq >= 0.25:
        myquat[0] = np.sqrt(q0sq)
        myquat[1] = (mya2[2] - mya3[1]) / (4.0 * myquat[0])
        myquat[2] = (mya3[0] - mya1[2]) / (4.0 * myquat[0])
        myquat[3] = (mya1[1] - mya2[0]) / (4.0 * myquat[0])
    elif q1sq >= 0.25:
        myquat[1] = np.sqrt(q1sq)
        myquat[0] = (mya2[2] - mya3[1]) / (4.0 * myquat[1])
        myquat[2] = (mya2[0] + mya1[1]) / (4.0 * myquat[1])
        myquat[3] = (mya1[2] + mya3[0]) / (4.0 * myquat[1])
    elif q2sq >= 0.25:
        myquat[2] = np.sqrt(q2sq)
        myquat[0] = (mya3[0] - mya1[2]) / (4.0 * myquat[2])
        myquat[1] = (mya2[0] + mya1[1]) / (4.0 * myquat[2])
        myquat[3] = (mya3[1] + mya2[2]) / (4.0 * myquat[2])
    elif q3sq >= 0.25:
        myquat[3] = np.sqrt(q3sq)
        myquat[0] = (mya1[1] - mya2[0]) / (4.0 * myquat[3])
        myquat[1] = (mya3[0] + mya1[2]) / (4.0 * myquat[3])
        myquat[2] = (mya3[1] + mya2[2]) / (4.0 * myquat[3])
    norm = 1.0 / np.sqrt(myquat[0] * myquat[0] + myquat[1] * myquat[1] + \
			  myquat[2] * myquat[2] + myquat[3] * myquat[3])
    myquat[0] *= norm
    myquat[1] *= norm
    myquat[2] *= norm
    myquat[3] *= norm

    return np.array([myquat[0], myquat[1], myquat[2], myquat[3]])


'''
description:  f1 modulation factor
return {*}
'''
@njit
def F1(  r,  eps,  a,  cut_0,  cut_lc,  cut_hc,  cut_lo,  cut_hi,  b_lo,  b_hi,  shift):
    #               8.0, 0.4, 0.2769075269685699, 0.7837754579041583, 0.34, 0.7, -126.24289692803721, -7.877076012859194,
    if (r > cut_hc):
        return 0.0
    elif (r > cut_hi):
        return eps * b_hi * (r-cut_hc) * (r-cut_hc)
    elif (r > cut_lo):
        tmp = 1 - np.exp(-(r-cut_0) * a)
        return eps * tmp * tmp - shift
    elif (r > cut_lc):
        return eps * b_lo * (r-cut_lc) * (r-cut_lc)
    else:
        return 0.0

def plot_F1(eps,  a,  cut_0,  cut_lc,  cut_hc,  cut_lo,  cut_hi,  b_lo,  b_hi,  shift):
    x = np.linspace(0,1,1000)
    y = [F1(i,eps,  a,  cut_0,  cut_lc,  cut_hc,  cut_lo,  cut_hi,  b_lo,  b_hi,  shift) for i in x]
    plt.figure(4,figsize=(8,6), dpi=300)
    plt.plot(x,y,)
    plt.savefig('F1_eps'+str(eps)+'_'+str(shift)+'.jpg')
    plt.close()



'''
description: f2 modulation factor
return {*}
'''
@njit
def F2(  r,  k,  cut_0,  cut_lc,  cut_hc,  cut_lo,  cut_hi,  b_lo,  b_hi,  cut_c):
    if (r < cut_lc or r > cut_hc):
        return 0.0
    elif (r < cut_lo):
        return k * b_lo * (cut_lc - r) * (cut_lc - r)
    elif (r < cut_hi):
        return k * 0.5 * ((r - cut_0) * (r - cut_0) - (cut_0 - cut_c) * (cut_0 - cut_c))
    else:
        return k * b_hi * (cut_hc - r) * (cut_hc - r)

'''
description: f3 modulation factor
return {*}
'''
@njit
def F3( rsq,  cutsq_ast,  cut_c,  lj1,  lj2,  eps,  b):
    if (rsq < cutsq_ast): 
        r2inv = 1.0 / rsq
        r6inv = r2inv * r2inv * r2inv
        return r6inv * (lj1 * r6inv - lj2)
    else:
        r = np.sqrt(rsq)
        rinv = 1.0 / r
        return eps * b * (cut_c - r) * (cut_c - r)

'''
description: f4 modulation factor
return {*}
'''
@njit
def F4( theta,  a,  theta_0,  dtheta_ast,  b,  dtheta_c):
    dtheta = theta-theta_0
    if (abs(dtheta) > dtheta_c):
        return 0.0
    elif (dtheta > dtheta_ast):
        return b * (dtheta-dtheta_c)*(dtheta-dtheta_c)
    elif (dtheta > -dtheta_ast):
        return 1 - a * dtheta*dtheta
    else:
        return b * (dtheta+dtheta_c)*(dtheta+dtheta_c)

def plot_F4(a,  theta_0,  dtheta_ast,  b,  dtheta_c):
    x = np.linspace(-3.14159265,3.14159265,1000)
    y = [F4(i,a,  theta_0,  dtheta_ast,  b,  dtheta_c) for i in x]
    plt.figure(4,figsize=(8,6), dpi=300)
    plt.plot(x,y)
    plt.xticks([-3.14159265,-3.14159265/2,0,3.14159265/2,3.14159265],['-180','-90','0','90','180'])
    plt.savefig('F4_eps'+str(a)+'.jpg')
    plt.close()


'''
description: f5 modulation factor
return {*}
'''
@njit
def F5( x,  a,  x_ast,  b,  x_c):
    if (x >= 0):
        return 1.0
    elif (x > x_ast):
        return 1 - a * x * x
    elif (x > x_c):
        return b * (x - x_c) * (x - x_c)
    else:
        return 0.0


'''
description: f6 modulation factor
return {*}
'''
@njit
def F6( theta,  a,  b):
    if (theta < b):
        return 0.0
    else:
        return 0.5 * a * (theta - b) * (theta - b)


def get_angle(a, b):
    """
    Get angle between a,b

    >>> a = [0, 1, 0]
    >>> b = [0, 0, 1]
    >>> round(get_angle(a,b),3)
    1.571

    """
    ab = np.dot(a, b)
    if ab > (1.-base.FLT_EPSILON): return 0
    elif ab < (-1.+base.FLT_EPSILON): return np.pi
    else: return np.arccos(ab)

def get_orthonormalized_base(v1, v2, v3):
    v1_norm2 = np.dot(v1, v1)
    v2_v1 = np.dot(v2, v1)

    v2 -= (v2_v1/v1_norm2) * v1

    v3_v1 = np.dot(v3, v1)
    v3_v2 = np.dot(v3, v2)
    v2_norm2 = np.dot(v2, v2)

    v3 -= (v3_v1/v1_norm2) * v1 + (v3_v2/v2_norm2) * v2

    v1 /= np.sqrt(v1_norm2)
    v2 /= np.sqrt(v2_norm2)
    v3 /= np.sqrt(np.dot(v3, v3))

    return v1, v2, v3

def get_random_vector_in_sphere(r=1):
    r2 = r*r
    v = np.random.uniform(-r, r, 3)

    while np.dot(v, v) > r2:
        v = np.random.uniform(-r, r, 3)

    return v

def get_random_vector():
    ransq = 1.

    while ransq >= 1.:
        ran1 = 1. - 2. * np.random.random()
        ran2 = 1. - 2. * np.random.random()
        ransq = ran1*ran1 + ran2*ran2

    ranh = 2. * np.sqrt(1. - ransq)
    return np.array([ran1*ranh, ran2*ranh, 1. - 2. * ransq])

def get_random_rotation_matrix():
    v1, v2, v3 = get_orthonormalized_base(get_random_vector(), get_random_vector(), get_random_vector())

    R = np.array([v1, v2, v3])
    # rotations have det == 1
    if np.linalg.det(R) < 0: R = np.array([v2, v1, v3])

    return R

def get_rotation_matrix(axis, anglest):
    """
    The argument anglest can be either an angle in radiants
    (accepted types are float, int or np.float64 or np.float64)
    or a tuple [angle, units] where angle a number and
    units is a string. It tells the routine whether to use degrees,
    radiants (the default) or base pairs turns

    axis --- Which axis to rotate about
        Ex: [0,0,1]
    anglest -- rotation in radians OR [angle, units]
        Accepted Units:
            "bp"
            "degrees"
            "radiants"
        Ex: [np.pi/2] == [np.pi/2, "radians"]
        Ex: [1, "bp"]

    """
    if not isinstance (anglest, (np.float64, np.float32, float, int)):
        if len(anglest) > 1:
            if anglest[1] in ["degrees", "deg", "o"]:
                angle = (np.pi / 180.) * anglest[0]
                #angle = np.deg2rad (anglest[0])
            elif anglest[1] in ["bp"]:
								# Notice that the choice of 35.9 DOES NOT correspond to the minimum free energy configuration.
								# This is usually not a problem, since the structure will istantly relax during simulation, but it can be
								# if you need a configuration with an equilibrium value for the linking number.
								# The minimum free energy angle depends on a number of factors (salt, temperature, length, and possibly more),
								# so if you need a configuration with 0 twist make sure to carefully choose a value for this angle
								# and force it in some way (e.g. by changing the angle value below to something else in your local copy).
                # Allow partial bp turns
                angle = float(anglest[0]) * (np.pi / 180.) * 35.9
                # Older versions of numpy don't implement deg2rad()
                #angle = int(anglest[0]) * np.deg2rad(35.9)
            else:
                angle = float(anglest[0])
        else:
            angle = float(anglest[0])
    else:
        angle = float(anglest) # in degrees, I think

    axis = np.array(axis)
    axis /= np.sqrt(np.dot(axis, axis))

    ct = np.cos(angle)
    st = np.sin(angle)
    olc = 1. - ct
    x, y, z = axis

    return np.array([[olc*x*x+ct, olc*x*y-st*z, olc*x*z+st*y],
                    [olc*x*y+st*z, olc*y*y+ct, olc*y*z-st*x],
                    [olc*x*z-st*y, olc*y*z+st*x, olc*z*z+ct]])

def norm(vec):
    return vec / np.sqrt(np.dot(vec,vec))

def get_base_spline(strand,circle_flag = -1 ,reverse = False):
    """
    return a cartesian spline that represents a fit through the bases for the strand 'strand'

    args:
    strand: base.Strand object
    circle_flag:True or False otherwise strand._circular
    """
    if circle_flag == -1:
        circle_flag = strand._circular

    base_pos = []
    for nuc in strand._nucleotides:
        base_pos.append(nuc.get_pos_base())

    if reverse:
        base_pos.reverse()

    if circle_flag:
        if reverse:
            base_pos.append(strand._nucleotides[-1].get_pos_base())
        else:
            base_pos.append(strand._nucleotides[0].get_pos_base())

    # interpolate bbms by interpolating each cartesian co-ordinate in turn
    xx = [vec[0] for vec in base_pos]
    yy = [vec[1] for vec in base_pos]
    zz = [vec[2] for vec in base_pos]
    # NB s = 0 forces interpolation through all data points
    spline_xx = scipy.interpolate.splrep(list(range(len(xx))), xx, k = 3, s = 0, per = circle_flag)
    spline_yy = scipy.interpolate.splrep(list(range(len(yy))), yy, k = 3, s = 0, per = circle_flag)
    spline_zz = scipy.interpolate.splrep(list(range(len(zz))), zz, k = 3, s = 0, per = circle_flag)
    return [spline_xx, spline_yy, spline_zz], (0, len(xx)-1)

def get_sayar_twist(s1, s2, smin, smax, npoints = 1000, circular = False, integral_type = "simple"):
    """
    return the twist for a given pair of spline fits, one through the bases of each strand

    from Sayar et al. 2010 Phys. Rev. E

    Just need to integrate along the contour parameter s that is common to both splines. We need the normalised tangent vector to the spline formed by the midpoint of the two input splines t(s), the normalised normal vector formed by the vectors between the splines u(s), and the derivative of the normalised normal vector between the splines d/ds (u(s)). NB, the normal u(s) vector should be orthogonal to the tangent vector t(s); we ensure this by using only the component orthogonal to t(s).

    Using integral_type = 'simple' and npoints = 200, it will give a correct twist, or at least one that gives a conserved linking number when combined with get_sayar_writhe

    args:
    s1: list of 3 splines corresponding to 3-D spline through strand 1's bases (e.g. use get_base_spline())
    s2: list of 3 splines corresponding to 3-D spline through strand 2's bases -- NB the splines should run in the same direction, i.e. one must reverse one of the splines if they come from get_base_spline (e.g. use get_base_spline(reverse = True))
    smin: minimum value for s, which parameterises the splines
    smax: maximum value for s, which parameterises the splines
    npoints: number of points for the discrete integration
    """

    s1xx, s1yy, s1zz = s1
    s2xx, s2yy, s2zz = s2

    # bpi is the base pair index parameter that common to both splines
    bpi = np.linspace(smin, smax, npoints)

    # find the midpoint between the input splines, as a function of base pair index
    mxx = (scipy.interpolate.splev(bpi, s1xx) + scipy.interpolate.splev(bpi, s2xx)) / 2
    myy = (scipy.interpolate.splev(bpi, s1yy) + scipy.interpolate.splev(bpi, s2yy)) / 2
    mzz = (scipy.interpolate.splev(bpi, s1zz) + scipy.interpolate.splev(bpi, s2zz)) / 2

    # contour_len[ii] is contour length along the midpoint curve of point ii
    delta_s = [np.sqrt((mxx[ii+1]-mxx[ii])**2+(myy[ii+1]-myy[ii])**2+(mzz[ii+1]-mzz[ii])**2) for ii in range(len(bpi)-1)]
    contour_len = np.cumsum(delta_s)
    contour_len = np.insert(contour_len, 0, 0)

    # ss is a linear sequence from first contour length element (which is 0) to last contour length element inclusive
    ss = np.linspace(contour_len[0], contour_len[-1], npoints)

    # get the midpoint spline as a function of contour length
    msxx = scipy.interpolate.splrep(contour_len, mxx, k = 3, s = 0, per = circular)
    msyy = scipy.interpolate.splrep(contour_len, myy, k = 3, s = 0, per = circular)
    mszz = scipy.interpolate.splrep(contour_len, mzz, k = 3, s = 0, per = circular)

    # find the tangent of the midpoint spline. 
    # the tangent t(s) is d/ds [r(s)], where r(s) = (mxx(s), myy(s), mzz(s)). So the tangent is t(s) = d/ds [r(s)] = (d/ds [mxx(s)], d/ds [myy(s)], d/ds [mzz(s)])
    # get discrete array of normalised tangent vectors; __call__(xxx, 1) returns the first derivative
    # the tangent vector is a unit vector
    dmxx = scipy.interpolate.splev(ss, msxx, 1)
    dmyy = scipy.interpolate.splev(ss, msyy, 1)
    dmzz = scipy.interpolate.splev(ss, mszz, 1)
    tt = list(range(len(ss)))
    for ii in range(len(ss)):
        tt[ii] = np.array([dmxx[ii], dmyy[ii], dmzz[ii]])

    # we also need the 'normal' vector u(s) which points between the base pairs. (or between the spline fits through the bases in this case)
    # n.b. these uxx, uyy, uzz are not normalised
    uxx_bpi = scipy.interpolate.splev(bpi, s2xx) - scipy.interpolate.splev(bpi, s1xx)
    uyy_bpi = scipy.interpolate.splev(bpi, s2yy) - scipy.interpolate.splev(bpi, s1yy)
    uzz_bpi = scipy.interpolate.splev(bpi, s2zz) - scipy.interpolate.splev(bpi, s1zz)

    # get the normal vector spline as a function of contour length
    suxx = scipy.interpolate.splrep(contour_len, uxx_bpi, k = 3, s = 0, per = circular)
    suyy = scipy.interpolate.splrep(contour_len, uyy_bpi, k = 3, s = 0, per = circular)
    suzz = scipy.interpolate.splrep(contour_len, uzz_bpi, k = 3, s = 0, per = circular)

    # evaluate the normal vector spline as a function of contour length
    uxx = scipy.interpolate.splev(ss, suxx)
    uyy = scipy.interpolate.splev(ss, suyy)
    uzz = scipy.interpolate.splev(ss, suzz)

    uu = list(range(len(ss)))
    for ii in range(len(ss)):
        uu[ii] = np.array([uxx[ii], uyy[ii], uzz[ii]])
        uu[ii] = uu[ii] - np.dot(tt[ii], uu[ii]) * tt[ii]
        # the normal vector should be normalised
        uu[ii] = norm(uu[ii])

    # and finally we need the derivatives of that vector u(s). It takes a bit of work to get a spline of the normalised version of u from the unnormalised one
    nuxx = [vec[0] for vec in uu]
    nuyy = [vec[1] for vec in uu]
    nuzz = [vec[2] for vec in uu]
    nusxx = scipy.interpolate.splrep(ss, nuxx, k = 3, s = 0, per = circular)
    nusyy = scipy.interpolate.splrep(ss, nuyy, k = 3, s = 0, per = circular)
    nuszz = scipy.interpolate.splrep(ss, nuzz, k = 3, s = 0, per = circular)
    duxx = scipy.interpolate.splev(ss, nusxx, 1)
    duyy = scipy.interpolate.splev(ss, nusyy, 1)
    duzz = scipy.interpolate.splev(ss, nuszz, 1)
    duu = list(range(len(ss)))
    for ii in range(len(ss)):
        duu[ii] = np.array([duxx[ii], duyy[ii], duzz[ii]])

    ds = float(contour_len[-1] - contour_len[0]) / (npoints - 1)
    # do the integration w.r.t. s
    if circular:
        srange = list(range(len(ss)-1))
    else:
        srange = list(range(len(ss)))
    if integral_type == "simple":
        integral = 0
        for ii in srange:
            #print np.dot(uu[ii], tt[ii])
            triple_scalar_product = np.dot(tt[ii], np.cross(uu[ii], duu[ii]))
            integral += triple_scalar_product * ds
    elif integral_type == "quad":
        assert False, "not currently supported; shouldn't be difficult to implement if wanted"
        integral, err = scipy.integrate.quad(twist_integrand, ss[0], ss[-1], args = (msxx, msyy, mszz, nusxx, nusyy, nuszz), limit = 500)
        print("error estimate:", err, file=sys.stderr)
        
    twist = integral/(2 * np.pi)

    return twist

@njit
def integral_speedup(circular,contour_len,ss,npoints,xx,yy,zz,dxx,dyy,dzz):
    integral = 0
    if circular:
        srange = list(range(len(ss)-1))
        ds = float(contour_len[-1] - contour_len[0]) / (npoints - 1)
    else:
        srange = list(range(len(ss)))
        ds = float(contour_len[-1] - contour_len[0]) / npoints
    for ii in srange:
        for jj in srange:
            # skip ii=jj and use symmetry in {ii, jj}
            if ii > jj:
                diff = np.array([xx[ii]-xx[jj], yy[ii] - yy[jj], zz[ii] - zz[jj]])
                diff_mag = np.sqrt(np.dot(diff, diff))
                diff_frac = diff / (diff_mag ** 3)
#                triple_scalar_product = np.dot(np.cross(tt[ii], tt[jj]), diff_frac)
#                temp = np.cross(tt[ii], tt[jj])
                temp = np.array([dyy[ii]*dzz[jj]-dzz[ii]*dyy[jj],dzz[ii]*dxx[jj]-dxx[ii]*dzz[jj],dxx[ii]*dyy[jj]-dyy[ii]*dxx[jj]])
                triple_scalar_product = np.dot(temp,diff_frac)
                integral += triple_scalar_product * ds * ds
    # multiply by 2 because we exploited the symmetry in {ii, jj} to skip half of the integral
    integral *= 2
    return integral

def get_sayar_writhe(splines1, smin, smax, splines2 = False, npoints = 1000, debug = False, circular = False, integral_type = "simple"):
    """
    return the writhe for a 3D spline fit through a set of duplex midpoints

    from Sayar et al. 2010 Phys. Rev. E

    Using integral_type = 'simple' and npoints = 200, it will give a correct writhe, or at least one that gives a conserved linking number when combined with get_sayar_twist

    args:
    splines1: list of 3 splines corresponding to either (if not splines2) a 3D spline through the duplex or (if splines2) strand 1's bases
    smin: minimum value for s, which parameterises the splines
    smax: maximum value for s, which parameterises the splines
    splines2: optionally, (see splines1) list of 3 splines corresponding to a 3D spline through strand2's bases
    npoints: number of points for the discrete integration
    debug: print a load of debugging information
    """
    import scipy.integrate

    # bpi is the base pair index parameter that common to both strands' splines
    bpi = np.linspace(smin, smax, npoints)

    ## get midpoint splines sxx, syy, szz
    if not splines2:
        # splines1 is the midpoint 3D spline as a function of base pair index
        sxx_bpi, syy_bpi, szz_bpi = splines1
        xx_bpi = scipy.interpolate.splev(bpi, sxx_bpi)
        yy_bpi = scipy.interpolate.splev(bpi, syy_bpi)
        zz_bpi = scipy.interpolate.splev(bpi, szz_bpi)
    else:
        # take splines1 and splines2 to be the splines through the bases of each strand; in that case we need to find the midpoint here first
        s1xx_bpi, s1yy_bpi, s1zz_bpi = splines1
        s2xx_bpi, s2yy_bpi, s2zz_bpi = splines2

        # find the midpoint as a function of base pair index between the input splines 
        xx_bpi = (scipy.interpolate.splev(bpi, s1xx_bpi) + scipy.interpolate.splev(bpi, s2xx_bpi)) / 2
        yy_bpi = (scipy.interpolate.splev(bpi, s1yy_bpi) + scipy.interpolate.splev(bpi, s2yy_bpi)) / 2
        zz_bpi = (scipy.interpolate.splev(bpi, s1zz_bpi) + scipy.interpolate.splev(bpi, s2zz_bpi)) / 2

    # contour_len[ii] is contour length along the midpoint curve of point ii
    delta_s = [np.sqrt((xx_bpi[ii+1]-xx_bpi[ii])**2+(yy_bpi[ii+1]-yy_bpi[ii])**2+(zz_bpi[ii+1]-zz_bpi[ii])**2) for ii in range(len(bpi)-1)]
    contour_len = np.cumsum(delta_s)
    contour_len = np.insert(contour_len, 0, 0)

    # ss is a linear sequence from first contour length element (which is 0) to last contour length element inclusive
    ss = np.linspace(contour_len[0], contour_len[-1], npoints)

    sxx = scipy.interpolate.splrep(contour_len, xx_bpi, k = 3, s = 0, per = circular)
    syy = scipy.interpolate.splrep(contour_len, yy_bpi, k = 3, s = 0, per = circular)
    szz = scipy.interpolate.splrep(contour_len, zz_bpi, k = 3, s = 0, per = circular)
    xx = scipy.interpolate.splev(ss, sxx)
    yy = scipy.interpolate.splev(ss, syy)
    zz = scipy.interpolate.splev(ss, szz)

    # find the tangent of the midpoint spline. 
    # the tangent t(s) is d/ds [r(s)], where r(s) = (mxx(s), myy(s), mzz(s)). So the tangent is t(s) = d/ds [r(s)] = (d/ds [mxx(s)], d/ds [myy(s)], d/ds [mzz(s)])
    # get discrete array of tangent vectors; __call__(xxx, 1) returns the first derivative
    dxx = scipy.interpolate.splev(ss, sxx, 1)
    dyy = scipy.interpolate.splev(ss, syy, 1)
    dzz = scipy.interpolate.splev(ss, szz, 1)
    tt = list(range(len(ss)))
    for ii in range(len(ss)):
        tt[ii] = np.array([dxx[ii], dyy[ii], dzz[ii]])

    # do the double integration w.r.t. s and s'

    if integral_type == "simple":
        integral = integral_speedup(circular = circular,contour_len = contour_len ,ss = ss ,npoints = npoints ,xx = xx,yy = yy,zz = zz,dxx =dxx,dyy=dyy,dzz=dzz)
    elif integral_type == "quad":
        pass
        #integral, err = scipy.integrate.quad(writhe_integrand2, ss[0], ss[-1], args = (sxx, syy, szz, ss[0], ss[-1]), limit = 100, epsabs = 1e-5, epsrel = 0)
    elif integral_type == "simps":
        srange = list(range(len(ss)))
        integrand = [[] for ii in srange]
        for ii in srange:
            for jj in srange:
                # skip ii=jj
                if ii == jj:
                    triple_scalar_product = 0
                else:
                    diff = np.array([xx[ii]-xx[jj], yy[ii] - yy[jj], zz[ii] - zz[jj]])
                    diff_mag = np.sqrt(np.dot(diff, diff))
                    diff_frac = diff / (diff_mag ** 3)
                    triple_scalar_product = np.dot(np.cross(tt[ii], tt[jj]), diff_frac)
                integrand[ii].append(triple_scalar_product)
        integral = scipy.integrate.simps(scipy.integrate.simps(integrand, ss), ss)
    else:
        assert False
        
    writhe = float(integral) / (4*np.pi)

    return writhe

def get_supercoiling_shape(contact,cutoff=7.5,minbandbp=25,missbarry=10):
    k,q = contact.shape
    looptop = {}    #key looptop value dict (key:(start,end),value,bandlen)
    bandlen = {}    #key bandlen value looptop
    maxbandlen = -1
    for i in range(k):
        loop = 1
        band_flag = 0 #last is close if 0 else !=0
        band = {}
        
        for j in range(1,k//2+1):
            if contact[(i+j)%k][i-j] < cutoff:
                if not band_flag:
                    loop += 1
                elif band_flag<=missbarry:
                    loop += band_flag + 1                    
                else:
                    if loop >= minbandbp:
                        # print(loop)
                        band[(j-band_flag-loop,j-band_flag-1)] = loop
                    loop = 1
                band_flag = 0
            else:
                band_flag +=1
        if loop >= minbandbp:
            band[(j+1-band_flag-loop,j+1-band_flag-1)] = loop
        if len(band) >0:
            for kk in band:
                if band[kk] in bandlen:
                    bandlen[band[kk]].append(i)
                else:
                    bandlen[band[kk]] = [i]
                maxbandlen = max(maxbandlen,band[kk])
            looptop[i] = dict(band)
    
    for kk in looptop:
        print(kk,looptop[kk])
    print(maxbandlen)
    print(bandlen[maxbandlen])
    mes = f'{maxbandlen} {bandlen[maxbandlen]}\n'
    return mes



def triad2(n1,n2): # matrix rotating <n-1> into <n> with respect to the cordinate system of the (n-1)-th triad
    #Supporting Information for \DNA elasticity from coarse-grained simulations: the effect of groove asymmetry"Enrico Skoruppa,1 Michiel Laleman,1 Stefanos K. Nomidis,1, 2 and Enrico Carlon1
    r = n1.cm_pos
    R = n2.cm_pos
    a3= n1._a3
    A3= n2._a3
    O = (r+R)/2                                        # mid point
    e3 = norm(a3-A3)                                   # twist/rise axis    
    y  = norm(r-R)
    e2 = norm(y-np.dot(y,e3)*e3)                       # roll/slide axis
    e1 = np.cross(e2,e3)                               # tilt/shift axis
    T = np.array([e1,e2,e3]).T                         
    return T



def get_loacl_twist(strand1,strand2,circle=False):
    # no pcb
    n = len(strand1._nucleotides)
    l2 = len(strand2._nucleotides)
    if n != l2:
        print('not the same lenth of two strands!')
        return False
    
    frame_matrix = [triad2(strand1._nucleotides[i],strand2._nucleotides[n-i-1]) for i in range(n)]
    rotation_matrix = [np.dot(frame_matrix[i].T,frame_matrix[i+1]) for i in range(n-1)] # Rij = Ri^-1 * Rj   (i=n-1,j=n)             R(n-1,n) = R(n-1).T * R(n) 
    if circle:
        rotation_matrix.append(np.dot(frame_matrix[-1].T,frame_matrix[0]))

    #euler[0] tilt [1] roll [2] twist
    local_twist = [Rotation.from_matrix(i).as_euler('xyz', degrees=True)[2] for i in rotation_matrix]
   

    return local_twist

@njit
def cal_cm_rg(pos_old,l_box,mass):
    n = len(pos_old)
    cm0 = pos_old[0].copy() #幸运粒子
    cm = np.zeros(3,dtype=np.double)
    pos = np.zeros((n,3),dtype=np.double)
    total_mass = np.sum(mass)
    for i in range(n):
        pos[i] = pos_old[i].copy()
        dr = pos[i]-cm0
        pos[i] -= l_box * np.rint (dr / l_box) #pos 距离cm0为盒子一半以内
        cm += pos[i]*mass[i]
    cm /= total_mass

    rg_sq = np.zeros(3,dtype=np.double)
    for i in range(n):
        rg_sq += (pos[i] - cm)*(pos[i] - cm)*mass[i]
    rg_sq /= total_mass

    return cm,rg_sq


def get_centerline(strand1,strand2=False,double_mode = False,l_box=False):
    n = len(strand1._nucleotides)
    centerline = np.zeros((n,3),dtype=np.double)
    if double_mode:
        try:
            for i in range(n):
                dr = strand1._nucleotides[i].cm_pos - strand2._nucleotides[n-1-i].cm_pos
                centerline[i] = (strand1._nucleotides[i].cm_pos - l_box * np.rint (dr / l_box) + strand2._nucleotides[n-1-i].cm_pos)/2
        except:
            print('strand2 or l_box not set')
            return False
        return centerline
    if strand2:
        for i in range(n):
            centerline[i] = strand1._nucleotides[i].pos_center_ds
    else:
        for i in range(n):
            centerline[i] = strand1._nucleotides[i].pos_back
    return centerline


def get_rgr_rgz(strand1,strand2=False,double_mode = False,l_box=False):
    _,rg_sq = cal_cm_rg(get_centerline(strand1,strand2,double_mode,l_box),l_box,mass=np.ones(strand1._n,dtype=int))
    rgr = math.sqrt(rg_sq[0]+rg_sq[1])
    rgz = math.sqrt(rg_sq[2])
#    rg = math.sqrt(rg_sq[0]+rg_sq[1]+rg_sq[2])
    return rgr ,rgz

def kde_2d(x, y, bandwidth=0.2, xbins=300, ybins=300, xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
    """
    计算二维KDE

    Parameters:
        x (numpy.ndarray): 输入数据的x坐标，一个一维数组
        y (numpy.ndarray): 输入数据的y坐标，一个一维数组
        bandwidth (float): 核函数的带宽参数
        xbins (int): x轴的bin数目
        ybins (int): y轴的bin数目
        xmin (float): x轴的最小值，如果为None，则默认为x中的最小值
        xmax (float): x轴的最大值，如果为None，则默认为x中的最大值
        ymin (float): y轴的最小值，如果为None，则默认为y中的最小值
        ymax (float): y轴的最大值，如果为None，则默认为y中的最大值
        **kwargs: 其他参数，用于传递给`gaussian_kde`函数

    Returns:
        grid (numpy.ndarray): 二维KDE的估计值，一个xbins x ybins的二维数组
        xedges (numpy.ndarray): x轴的bin边界，一个长度为xbins+1的一维数组
        yedges (numpy.ndarray): y轴的bin边界，一个长度为ybins+1的一维数组
    """
    # 将输入数据转换为一个二维矩阵
    data = np.vstack([x, y])
    # 计算x轴和y轴的bin边界
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    if ymin is None:
        ymin = y.min()
    if ymax is None:
        ymax = y.max()
    xedges = np.linspace(xmin, xmax, xbins + 1)
    yedges = np.linspace(ymin, ymax, ybins + 1)

    # 构造一个二维网格，用于计算KDE
    xx, yy = np.meshgrid(xedges, yedges)
    xy_grid = np.vstack([xx.ravel(), yy.ravel()])

    # 计算二维KDE的估计值
    # kde = gaussian_kde(data, bw_method=bandwidth, **kwargs)
    kde = gaussian_kde(data, **kwargs)
    z = kde(xy_grid)
    # z = z / (z.sum() * np.diff(xy_grid[:2]).prod())
    grid = z.reshape(xx.shape)
    return grid, xedges, yedges



def kde_1d(x, bandwidth=0.2, num_points=1000, xmin=None, xmax=None, **kwargs):
    """
    计算一维KDE

    Parameters:
        x (numpy.ndarray): 输入数据，一个一维数组
        bandwidth (float): 核函数的带宽参数
        num_points (int): 估计KDE时使用的点的数量
        xmin (float): x轴的最小值，如果为None，则默认为x中的最小值
        xmax (float): x轴的最大值，如果为None，则默认为x中的最大值
        **kwargs: 其他参数，用于传递给`gaussian_kde`函数

    Returns:
        x_values (numpy.ndarray): 估计的x轴的值，一个一维数组
        kde_values (numpy.ndarray): 估计的KDE函数值，一个一维数组
    """
    # 计算x轴的取值范围
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x_values = np.linspace(xmin, xmax, num_points+1)

    # 计算KDE函数
    kde = gaussian_kde(x, **kwargs)
    kde_values = kde(x_values)

    return x_values, kde_values

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def kde_sklearn(x, kernel='gaussian', grid_points=1000, xmin=None, xmax=None,cv=5):
    """Calculate the kernel density estimate of a 1D array of data using Scikit-learn.

    Args:
        x (array-like): 1D array of data to compute KDE for.
        kernel (str, optional): The kernel function to use. Must be one of ['gaussian', 'tophat', 'epanechnikov', 
                                 'exponential', 'linear', 'cosine']. Defaults to 'gaussian'.
        grid_points (int, optional): The number of points to use in the grid for evaluating the KDE. Defaults to 1000.
        cv (int, optional): The number of cross-validation folds to use when performing the grid search for 
                            bandwidth selection. Defaults to 5.

    Returns:
        tuple: A tuple containing the x and y values of the KDE curve.
    """
    # Define the kernel density estimator
    kde = KernelDensity(kernel=kernel)

    # Define the parameter grid to search over
    param_grid = {'bandwidth': np.logspace(-1, 1, 20)}

    # Perform the grid search to find the optimal bandwidth
    grid_search = GridSearchCV(kde, param_grid=param_grid, cv=cv)
    grid_search.fit(x[:, np.newaxis])
    bandwidth = grid_search.best_params_['bandwidth']

    # Fit the KDE with the optimal bandwidth
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(x[:, np.newaxis])

    # Evaluate the KDE on a grid of points
        # 计算x轴的取值范围
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x_grid = np.linspace(xmin, xmax, grid_points+1)
    log_prob = kde.score_samples(x_grid[:, np.newaxis])
    y = np.exp(log_prob)

    return x_grid, y

@njit
def get_supercoiling_band(contact,cutoff=7.5):
    k,q = contact.shape
    band_array = np.zeros(k,dtype=np.int32)
    for i in range(29,k-19):
        tem_array = np.zeros(k-i,dtype=np.int32)
        temleft = 0
        temright = i
        for j in range(k-i):
            if contact[j,i+j] < cutoff:
                tem_array[j] = 1
        temleft,temright  = find_longest_subarray(tem_array,2)
        if 9 < (temright - temleft + 1) < 51:
            band_array[temleft:temright+1] += 1
            band_array[i+temleft:i+temright+1] += 1
    
    return np.where(band_array > 19.9, 1, 0)

@njit
def find_longest_subarray(arr, k):
    n = len(arr)
    left, right = 0, 0
    zero_count = 0
    max_len = 0
    subarray_left, subarray_right = -1, -2
    # sub_percentaget = 0
    for i in range(n):
        if arr[i] == 0:
            zero_count += 1
            continue
        
        if arr[i+1] == 0 or i == n-1:
            right = i
            while zero_count > k:
                if arr[left] == 0:
                    zero_count -= 1
                left += 1
            while arr[left] == 0:
                zero_count -= 1
                left += 1
            if right - left + 1 > max_len:
                max_len = right - left + 1
                subarray_left, subarray_right = left, right
                # sub_percentaget = max_len - zero_count
    return subarray_left, subarray_right #, sub_percentaget


                

if __name__ == "__main__":
    a = np.array([0,0,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1])
    l,r=find_longest_subarray(a,2)
    print(l,r )
