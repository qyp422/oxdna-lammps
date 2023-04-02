'''
Author: qyp422
Date: 2023-03-23 20:46:00
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-03-26 17:30:50
Description: 

Copyright (c) 2023 by qyp422, All Rights Reserved. 
'''
import os,sys,datetime,time
import base
import mathfuction as mf
import dumpreader as dr
import numpy as np
import matplotlib.pyplot as plt

def main(argv_name,argv_top,lk_n=False):
    start=datetime.datetime.now()
    
    filename = os.path.basename(argv_name)
    topfile = os.path.basename(argv_top)
    try:
        lk_n = int(lk_n)
    except:
        lk_n = 32

    rg_r = [] #存放该方向rg
    rg_z = [] #z方向rg
    pos_z = []
    
    # open file
    r = dr.Lammps_dumpreader(filename,topfile)
    # w = base.File_system(pwd,filename)
    # w._lk = lk_n
    n_probe = r._system.probe_mol
    double = False
    if r._system.target_mol:
        double = True

    frequency = 4000 # every times of f will output

    condition = True #frame 0 always read
    
    while r._read_single_frame(skip = not condition):
        
        
        if condition:
            
            t = r._system._time

            
            #sor

            # w.add_sor_message(t,r._system.sor)


            #rg no pre
            if double:
                for i in range(n_probe):
                    temrg_r,temrg_z = mf.get_rgr_rgz(r._system._strands[i],r._system._strands[i+n_probe],double_mode=double,l_box=r._system.l_box)
                    rg_r.append(temrg_r)
                    rg_z.append(temrg_z)
            else:
                for i in range(n_probe):
                    temrg_r,temrg_z = mf.get_rgr_rgz(r._system._strands[i],double_mode=double,l_box=r._system.l_box)
                    rg_r.append(temrg_r)
                    rg_z.append(temrg_z)


            pos_z += r._system.get_i_density_distribution(i = 2)
            #ovt
            # w.add_ovt_output(r._system.ovt_lammps_output)
        
        # reader condition
        condition = True # if r.frame_num % frequency == 0 else False
    # close file

    # r._system.get_system_graph(path=pwd,filename = filename)

    # w.add_kde_message(t,r._system.get_i_density_distribution(i = 2))
    


    # del w
    del r
    end=datetime.datetime.now()
    print(f'start at {start}')
    print(f'end   at {end}')
    print(f'total wall time is  {end-start}')
    print(f'filename {filename} done!')
    
    rg = np.vstack((np.array(rg_r),np.array(rg_z)))
    print(rg.shape)
    return rg,pos_z

    
if __name__ == "__main__":
    start=datetime.datetime.now()
    num_names = [0,1,2,3]
    filenames = [f'original_T_300_{i}.lammpstrj' for i in num_names ]

    current_dir = os.getcwd()
    folder_name = os.path.basename(current_dir)
    rg_array = []
    pos_z_array = []
    for f in filenames:
        t_rg,t_pos_z = main(f,'top.data')
        rg_array.append(t_rg)
        pos_z_array += t_pos_z
    rg_array = np.concatenate(rg_array,axis=1)
    pos_z_array = np.array(pos_z_array)

    np.savetxt(f'{folder_name}_rg_rz.txt',rg_array.T,fmt='%.6f')
    np.savetxt(f'{folder_name}_pos_z.txt',pos_z_array,fmt='%.6f',newline=' ')

    print('start plot')
    kde, xedges, yedges = mf.kde_2d(rg_array[0], rg_array[1],xmin=0.,ymin=0.,xmax=4,ymax=4)
    np.savetxt(f'{folder_name}_rg_array_plot.txt',kde,fmt='%.6f')
    fig, ax = plt.subplots()



    pcm = ax.pcolormesh(xedges, yedges, kde, cmap=plt.cm.gist_earth_r,vmin=0,vmax=1)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(f'{folder_name}_rg_rz_1.pdf',dpi=300)
    plt.close()
    print('end plot')
    print('start plot')

    x, kde_v = mf.kde_1d(pos_z_array, num_points=2000, xmin=0,xmax=20,bw_method=0.1)
    fig, ax = plt.subplots()

    pcm = ax.plot(x,kde_v,'k-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(f'{folder_name}_pos_z_{0.1}.pdf',dpi=300)
    plt.close()
    print('end plot')

    rg_r = np.mean(rg_array[0])
    rg_z = np.mean(rg_array[1])
    rg_rms_r = np.sqrt(np.mean(rg_array[0]**2))
    rg_rms_z = np.sqrt(np.mean(rg_array[1]**2))

    z_mean = np.mean(pos_z_array)
    
    # 打开文件
    with open(f'{folder_name}_mean_rg.txt', 'w') as file:
    # 写入数据
        file.write(f'rg_r = {rg_r}\n')
        file.write(f'rg_z = {rg_z}\n')
        file.write(f'rg_rms_r = {rg_rms_r}\n')
        file.write(f'rg_rms_z = {rg_rms_z}\n')
        file.write(f'z_mean = {z_mean}\n')



    end=datetime.datetime.now()
    print(f'start at {start}')
    print(f'end   at {end}')
    print(f'total wall time is  {end-start}')