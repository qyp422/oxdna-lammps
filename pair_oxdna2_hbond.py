'''
Author: qyp422
Date: 2022-10-13 15:29:48
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2022-11-04 19:23:01

Description: 
hydrogen-bonding interaction between complementary
F1 
8.0, 0.4, 0.2769075269685699, 0.7837754579041583, 0.34, 0.7, -126.24289692803721, -7.877076012859194,
Copyright (c) 2022 by qyp422, All Rights Reserved. 
'''
try:
    from . import mathfuction as mf
except:
    import mathfuction as mf

import numpy as np
import math
from numba import njit
import base

print('\n' + __file__ + ' is called\n')

# initialize 

# hb parameters in lammps
# pair_coeff 1 4 oxdna2/hbond   seqdep 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
# pair_coeff 2 3 oxdna2/hbond   seqdep 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
initialize_flag = True
if initialize_flag:
    epsilon_hb_one = 1.0678
    a_hb_one = 8.0
    cut_hb_0_one = 0.4
    cut_hb_c_one = 0.75
    cut_hb_lo_one = 0.34
    cut_hb_hi_one = 0.7

    a_hb1_one = 1.5
    theta_hb1_0_one = 0.0
    dtheta_hb1_ast_one = 0.7

    a_hb2_one = 1.5
    theta_hb2_0_one = 0.0
    dtheta_hb2_ast_one = 0.7

    a_hb3_one = 1.5
    theta_hb3_0_one = 0.0
    dtheta_hb3_ast_one = 0.7

    a_hb4_one = 0.46
    theta_hb4_0_one = 3.141592653589793
    dtheta_hb4_ast_one = 0.7

    a_hb7_one = 4.0
    theta_hb7_0_one = 1.5707963267948966
    dtheta_hb7_ast_one = 0.45

    a_hb8_one = 4.0
    theta_hb8_0_one = 1.5707963267948966
    dtheta_hb8_ast_one = 0.45

    b_hb_lo_one = 2*a_hb_one*np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one))*\
    2*a_hb_one*np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one))*\
    (1-np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one)))/\
    (4*((1-np.exp(-a_hb_one*(cut_hb_lo_one -cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one)))-\
    (1-np.exp(-a_hb_one*(cut_hb_c_one -cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_c_one-cut_hb_0_one)))))


    cut_hb_lc_one = cut_hb_lo_one - a_hb_one*np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one))*\
    (1-np.exp(-a_hb_one*(cut_hb_lo_one-cut_hb_0_one)))/b_hb_lo_one

    b_hb_hi_one = 2*a_hb_one*np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one))*\
    2*a_hb_one*np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one))*\
    (1-np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one)))/\
    (4*((1-np.exp(-a_hb_one*(cut_hb_hi_one -cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one)))-\
    (1-np.exp(-a_hb_one*(cut_hb_c_one -cut_hb_0_one)))*\
    (1-np.exp(-a_hb_one*(cut_hb_c_one-cut_hb_0_one)))))

    cut_hb_hc_one = cut_hb_hi_one - a_hb_one*np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one))*\
    (1-np.exp(-a_hb_one*(cut_hb_hi_one-cut_hb_0_one)))/b_hb_hi_one
    
    tmp = 1 - np.exp(-(cut_hb_c_one-cut_hb_0_one) * a_hb_one)
    shift_hb_one = epsilon_hb_one * tmp * tmp

    b_hb1_one = a_hb1_one*a_hb1_one*dtheta_hb1_ast_one*dtheta_hb1_ast_one/(1-a_hb1_one*dtheta_hb1_ast_one*dtheta_hb1_ast_one)
    dtheta_hb1_c_one = 1/(a_hb1_one*dtheta_hb1_ast_one)


    b_hb2_one = a_hb2_one*a_hb2_one*dtheta_hb2_ast_one*dtheta_hb2_ast_one/(1-a_hb2_one*dtheta_hb2_ast_one*dtheta_hb2_ast_one)
    dtheta_hb2_c_one = 1/(a_hb2_one*dtheta_hb2_ast_one)

    b_hb3_one = a_hb3_one*a_hb3_one*dtheta_hb3_ast_one*dtheta_hb3_ast_one/(1-a_hb3_one*dtheta_hb3_ast_one*dtheta_hb3_ast_one)
    dtheta_hb3_c_one = 1/(a_hb3_one*dtheta_hb3_ast_one)

    b_hb4_one = a_hb4_one*a_hb4_one*dtheta_hb4_ast_one*dtheta_hb4_ast_one/(1-a_hb4_one*dtheta_hb4_ast_one*dtheta_hb4_ast_one)
    dtheta_hb4_c_one = 1/(a_hb4_one*dtheta_hb4_ast_one)

    b_hb7_one = a_hb7_one*a_hb7_one*dtheta_hb7_ast_one*dtheta_hb7_ast_one/(1-a_hb7_one*dtheta_hb7_ast_one*dtheta_hb7_ast_one)
    dtheta_hb7_c_one = 1/(a_hb7_one*dtheta_hb7_ast_one)

    b_hb8_one = a_hb8_one*a_hb8_one*dtheta_hb8_ast_one*dtheta_hb8_ast_one/(1-a_hb8_one*dtheta_hb8_ast_one*dtheta_hb8_ast_one)
    dtheta_hb8_c_one = 1/(a_hb8_one*dtheta_hb8_ast_one)
#epsilon_hb setting
    epsilon_hb = np.zeros((4,4))
    shift_hb = np.zeros((4,4))
    alpha = np.array([[1.00000,1.00000,1.00000,0.82915],
    [1.00000,1.00000,1.15413,1.00000],
    [1.00000,1.15413,1.00000,1.00000],
    [0.82915,1.00000,1.00000,1.00000]])

    for i in range(4):
        for j in range(4):
            shift_hb[i][j] = shift_hb_one * alpha[i][j]
            epsilon_hb[i][j] = epsilon_hb_one * alpha[i][j]

    factor_lj = 1.0 # lj unit
    
    initialize_flag = False

print("\npair_oxdna2_hbond seqdepflag=1 initialize done\n")




'''
description: 
param {np} a_pos    particle a cm pos
param {np} ax       particle a a1
param {np} az       particle a a3
param {int} a_base  particle a type(0-7)
param {np} b_pos
param {np} bx
param {np} bz
param {np} b_base
return {float}
'''
@njit
def hb_energy(a_pos:np.array,ax:np.array,az:np.array,a_base:int,b_pos:np.array,bx:np.array,bz:np.array,b_base:int,box: np.array  = np.array([10000,10000,10000]) ,seqdep : bool = True , periodic : bool = True ) -> float:
    energy = 0.0
# Nucleotide.base A:0 C:1 G:2 T:3
# for seqave use a a 
    if seqdep:
        if (a_base + b_base) % 4 != 3:
            return energy
        type_a = a_base % 4
        type_b = b_base % 4
        
    else:
        type_a = a_base % 4
        type_b = a_base % 4
#distance COM-hbonding site
    

    ra_chb = np.zeros(3)
    rb_chb = np.zeros(3)
    delr_hb = np.zeros(3)
    delr_hb_norm = np.zeros(3)
    d_chb = base.POS_BASE

# for periodic boundary
    if periodic:
        diff = box * np.rint((a_pos - b_pos)/ box)
        diff[2] = 0.
    else:
        diff = np.zeros(3)
#quaternions and Cartesian unit vectors in lab frame
    # ax = a._a1
    # az = a._a3
    # bx = b._a1
    # bz = b._a3
#vectors COM-h-bonding site in lab frame
    # ra_chb[0] = d_chb*ax[0]
    # ra_chb[1] = d_chb*ax[1]
    # ra_chb[2] = d_chb*ax[2]
    # rb_chb[0] = d_chb*bx[0]
    # rb_chb[1] = d_chb*bx[1]
    # rb_chb[2] = d_chb*bx[2]
    ra_chb = d_chb*ax
    rb_chb = d_chb*bx
# period
# unwrapped atom coordinates
#vector h-bonding site b to a


    delr_hb = a_pos + ra_chb - b_pos - rb_chb - diff
    # delr_hb[0] = a[0] + ra_chb[0] - b[0] - rb_chb[0]
    # delr_hb[1] = a[1] + ra_chb[1] - b[1] - rb_chb[1]
    # delr_hb[2] = a[2] + ra_chb[2] - b[2] - rb_chb[2]    
    rsq_hb = delr_hb[0]*delr_hb[0] + delr_hb[1]*delr_hb[1] + delr_hb[2]*delr_hb[2]
    r_hb = np.sqrt(rsq_hb)
    if (r_hb):
        rinv_hb = 1.0/r_hb
    else:
        return 1000
    # delr_hb_norm[0] = delr_hb[0] * rinv_hb
    # delr_hb_norm[1] = delr_hb[1] * rinv_hb
    # delr_hb_norm[2] = delr_hb[2] * rinv_hb
    delr_hb_norm = delr_hb * rinv_hb
    f1 = mf.F1(r_hb, epsilon_hb[type_a][type_b],a_hb_one,cut_hb_0_one,cut_hb_lc_one,\
    cut_hb_hc_one,cut_hb_lo_one,cut_hb_hi_one,b_hb_lo_one,b_hb_hi_one,shift_hb[type_a][type_b])
    if (f1):
        cost1 = -1.0*np.dot(ax,bx)
        if (cost1 >  1.0):
            cost1 =  1.0
        if (cost1 < -1.0):
            cost1 = -1.0
        theta1 = math.acos(cost1)

        f4t1 = mf.F4(theta1, a_hb1_one, theta_hb1_0_one, dtheta_hb1_ast_one,\
        b_hb1_one, dtheta_hb1_c_one)

        if (f4t1):

            cost2 = -1.0*np.dot(ax,delr_hb_norm)
            if (cost2 >  1.0):
                cost2 =  1.0
            if (cost2 < -1.0):
                cost2 = -1.0
            theta2 = math.acos(cost2)

            f4t2 = mf.F4(theta2, a_hb2_one, theta_hb2_0_one, dtheta_hb2_ast_one,\
            b_hb2_one, dtheta_hb2_c_one)

            if (f4t2) :

                cost3 = np.dot(bx,delr_hb_norm)
                if (cost3 >  1.0) :
                    cost3 =  1.0
                if (cost3 < -1.0) :
                    cost3 = -1.0
                theta3 = math.acos(cost3)

                f4t3 = mf.F4(theta3, a_hb3_one, theta_hb3_0_one, dtheta_hb3_ast_one,\
                b_hb3_one, dtheta_hb3_c_one)


                if (f4t3) :

                    cost4 = np.dot(az,bz)
                    if (cost4 >  1.0) :
                        cost4 =  1.0
                    if (cost4 < -1.0) :
                        cost4 = -1.0
                    theta4 = math.acos(cost4)

                    f4t4 = mf.F4(theta4, a_hb4_one, theta_hb4_0_one, dtheta_hb4_ast_one,\
                    b_hb4_one, dtheta_hb4_c_one)


                    if (f4t4) :

                        cost7 = -1.0*np.dot(az,delr_hb_norm)
                        if (cost7 >  1.0) :
                            cost7 =  1.0
                        if (cost7 < -1.0) :
                            cost7 = -1.0
                        theta7 = math.acos(cost7)

                        f4t7 = mf.F4(theta7, a_hb7_one, theta_hb7_0_one, dtheta_hb7_ast_one,\
                        b_hb7_one, dtheta_hb7_c_one)


                        if (f4t7) :

                            cost8 = np.dot(bz,delr_hb_norm)
                            if (cost8 >  1.0) :
                                cost8 =  1.0
                            if (cost8 < -1.0) :
                                cost8 = -1.0
                            theta8 = math.acos(cost8)

                            f4t8 = mf.F4(theta8, a_hb8_one, theta_hb8_0_one, dtheta_hb8_ast_one,\
                            b_hb8_one, dtheta_hb8_c_one)

                            energy = f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj
    return energy

def plot_hb_function():
    import matplotlib.pyplot as plt
    plt.rc('font', family = 'Arial',size = 30)
    plt.figure(4,figsize=(8*2,6), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplots_adjust(left=0.1,bottom=0.15,right=0.95,top=0.90,wspace=0.3,hspace=0.2)
    plt.subplot(121)
    
    x = np.linspace(0,1,1000)
    y = [mf.F1(i,1.0678, 8.0, 0.4, 0.2769075269685699, 0.7837754579041583, 0.34, 0.7, -126.24289692803721, -7.877076012859194, 0.9418826091340452) for i in x]
    c = 0.88537
    y1 = [mf.F1(i,1.0678*c, 8.0, 0.4, 0.2769075269685699, 0.7837754579041583, 0.34, 0.7, -126.24289692803721, -7.877076012859194, 0.9418826091340452*c) for i in x]
    c = 1.15413
    y2 = [mf.F1(i,1.0678*c, 8.0, 0.4, 0.2769075269685699, 0.7837754579041583, 0.34, 0.7, -126.24289692803721, -7.877076012859194, 0.9418826091340452*c) for i in x]
    cxline  = 'violet'
    lw = 2
    t1 = 0.2769075269685699
    t2 = 0.7837754579041583
    plt.plot(x,y,'r-',label='seqav')
    plt.plot(x,y1,'g-',label='seqAT')
    plt.plot(x,y2,'b-',label='seqCG')
    plt.vlines(t1, -1.3, 0.1,color=cxline,linewidth=lw,linestyles='dashed')
    plt.vlines(t2, -1.3, 0.1,color=cxline,linewidth=lw,linestyles='dashed')
    plt.xlabel('r/'+chr(963))
    plt.legend(prop={'family':'Arial', 'size':22})
    plt.ylim(-1.4,0.2)
    plt.xticks([0,t1,0.5,t2,1],['0','0.277','0.5','0.784','1'])    

    
    plt.subplot(122)
    x = np.linspace(-3.14159265,3.14159265,1000)
    y = [mf.F4(i,1.5,0.0,0.7,4.16038,0.952381) for i in x]
    y1 = [mf.F4(i,0.46,3.1415926,0.7,0.133855,3.10559) for i in x]
    y2 = [mf.F4(i,4.0,3.1415926/2,0.45,17.0526,0.555556) for i in x]
    t1 = np.array([-0.952381,0.952381,np.pi/2-0.555556,np.pi/2+0.555556])
    plt.vlines(t1, 0, 1,color=cxline,linewidth=lw,linestyles='dashed')

    # plt.text(t1[0],1.1,'-54.567',verticalalignment='center',horizontalalignment='center')

    plt.plot(x,y,'r-',label=chr(952)+'$_1$/'+chr(952)+'$_2$/'+chr(952)+'$_3$')
    plt.plot(x,y1,'g-',label=chr(952)+'$_4$')
    plt.plot(x,y2,'b-',label=chr(952)+'$_7$/'+chr(952)+'$_8$')
    plt.xlabel(chr(952))
    plt.xticks([-3.14159265,-3.14159265/2,0,3.14159265/2,3.14159265],['-180','-90','0','90','180'])    
    plt.legend(prop={'family':'Arial', 'size':18})
    plt.savefig('h_b.jpg')
    plt.close()


if __name__ == "__main__":
    import datetime
    import time
    start=datetime.datetime.now()
    s = time.time()
    print((str(start))) 
#text
    ap = np.array([-11.6529995456132,40.9976721892332,1.95644471904994])
    ax = np.array([ 0.580902314684835,-0.675712388535067,0.453833966086748])
    az = np.array([-0.440472726822448,0.207911682216009,0.873359324289676]) 
    bp = np.array([-10.9949620289164,40.2509456195147,2.55711719231319])
    bx = np.array([ -0.622809552969655,0.508163631724362,-0.594876444417186]) 
    bz = np.array([ 0.449944227309256,-0.389384741475287,-0.803697527318147]) 
    # plot_hb_function()
    for i in range(10):
        print(hb_energy(ap,ax,az,0,bp,bx,bz,3,periodic = False))

#textend
    e = time.time()
    end=datetime.datetime.now()
    print((str(end)))
    print('# Done                                ')     
    h = end-start
    print((str(h)))
    print('take ' +str(datetime.timedelta(seconds=(e-s))) + ' second')


    