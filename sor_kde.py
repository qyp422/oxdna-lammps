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

    sor_p = [] #存放probe的sor
    sor_p_d = [] #角度
    sor_t = [] #存放杂交的target的sor
    sor_t_d = [] #角度
    pos_z = [] #probe的z和杂交的z
    
    # open file
    r = dr.Lammps_dumpreader(filename,topfile)
    # w = base.File_system(pwd,filename)
    # w._lk = lk_n
    n_probe = r._system.probe_mol
    double = False
    if r._system.target_mol:
        double = True

    frequency = 4000 # every times of f will output

    condition = False #frame 0 always read
    
    while r._read_single_frame(skip = not condition):
        
        
        if condition:
            
            t = r._system._time

            target_set = set()

            #find hb
            r._system.get_hb_pair(mode = 'search')
            #sor
            for i in range(n_probe):
                for n in r._system._strands[i]._nucleotides:
                    if np.abs(n._a3[2])>1:
                        sor_p.append(1.0)
                        sor_p_d.append(np.degrees(np.arcsin(1)))
                    else:
                        sor_p.append(1.5 * n._a3[2] * n._a3[2] - 0.5)
                        sor_p_d.append(np.degrees(np.arcsin(np.abs(n._a3[2]))))
                    pos_z.append(n.cm_pos[2])
                if r._system._strands[i].H_interactions:
                    target_set.update(r._system._strands[i].H_interactions.keys())
            
            for i in target_set:
                for n in r._system._strands[i]._nucleotides:
                    if np.abs(n._a3[2])>1:
                        sor_t.append(1.0)
                        sor_t_d.append(np.degrees(np.arcsin(1)))
                    else:
                        sor_t.append(1.5 * n._a3[2] * n._a3[2] - 0.5)
                        sor_t_d.append(np.degrees(np.arcsin(np.abs(n._a3[2]))))
                    pos_z.append(n.cm_pos[2])
            

            #ovt
            # w.add_ovt_output(r._system.ovt_lammps_output)
        
        # reader condition
        condition = True  if r.frame_num >= 3500 else False
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
    
    return sor_p,sor_t,sor_p_d,sor_t_d,pos_z

def plot_1d(x,kde_v,folder_name,file_name):
    
    fig, ax = plt.subplots()

    pcm = ax.plot(x,kde_v,'k-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(f'{folder_name}_{file_name}.pdf',dpi=300)
    plt.close()
    print(f'end plot {folder_name}_{file_name}.pdf')

if __name__ == "__main__":
    start=datetime.datetime.now()
    num_names = [0,1,2,3]
    filenames = [f'original_T_300_{i}.lammpstrj' for i in num_names ]

    current_dir = os.getcwd()
    folder_name = os.path.basename(current_dir)
    sor_p_array = []
    sor_t_array = []
    sor_p_d_array = []
    sor_t_d_array = []
    pos_z_array = []
    for f in filenames:
        tem = main(f,'top.data')
        sor_p_array += tem[0]
        sor_t_array += tem[1]
        sor_p_d_array += tem[2]
        sor_t_d_array += tem[3]
        pos_z_array += tem[4]

    sor_p_array   = np.array(sor_p_array)
    sor_t_array   = np.array(sor_t_array)
    sor_p_d_array = np.array(sor_p_d_array)
    sor_t_d_array = np.array(sor_t_d_array)
    pos_z_array = np.array(pos_z_array)

    np.savetxt(f'{folder_name}_sor_p_array.txt',sor_p_array  ,fmt='%.6f',newline=' ')
    np.savetxt(f'{folder_name}_sor_t_array.txt',sor_t_array  ,fmt='%.6f',newline=' ')
    np.savetxt(f'{folder_name}_sor_p_d_array.txt',sor_p_d_array ,fmt='%.6f',newline=' ')
    np.savetxt(f'{folder_name}_sor_t_d_array.txt',sor_t_d_array ,fmt='%.6f',newline=' ')
    np.savetxt(f'{folder_name}_pos_z_array.txt',pos_z_array,fmt='%.6f',newline=' ')

    mean_sor_p_array   = np.mean(sor_p_array  )
    mean_sor_t_array   = np.mean(sor_t_array  )
    mean_sor_p_d_array = np.mean(sor_p_d_array)
    mean_sor_t_d_array = np.mean(sor_t_d_array)
    z_mean = np.mean(pos_z_array)
    
    # 打开文件
    with open(f'{folder_name}_mean_sor.txt', 'w') as file:
    # 写入数据
        file.write(f'mean_sor_p_array = {mean_sor_p_array}\n')
        file.write(f'mean_sor_t_array = {mean_sor_t_array}\n')
        file.write(f'mean_sor_p_d_array = {mean_sor_p_d_array}\n')
        file.write(f'mean_sor_t_d_array = {mean_sor_t_d_array}\n')
        file.write(f'z_mean = {z_mean}\n')

    try:
        x, kde_v = mf.kde_1d(sor_p_array, num_points=1000, xmin=-1,xmax=1,bw_method=0.1)
        plot_1d(x,kde_v,folder_name,file_name=f'sor_p_array')
    except:
        pass

    try:
        x, kde_v = mf.kde_1d(sor_t_array, num_points=1000, xmin=-1,xmax=1,bw_method=0.1)
        plot_1d(x,kde_v,folder_name,file_name=f'sor_t_array')
    except:
        pass

    try:
        x, kde_v = mf.kde_1d(sor_p_d_array, num_points=1000, xmin=0,xmax=90,bw_method=0.1)
        plot_1d(x,kde_v,folder_name,file_name=f'sor_p_d_array')
    except:
        pass

    try:
        x, kde_v = mf.kde_1d(sor_t_d_array, num_points=1000, xmin=0,xmax=90,bw_method=0.1)
        plot_1d(x,kde_v,folder_name,file_name=f'sor_t_d_array')
    except:
        pass
    
    try:
        x, kde_v = mf.kde_1d(pos_z_array, num_points=2000, xmin=0,xmax=20,bw_method=0.1)
        plot_1d(x,kde_v,folder_name,file_name=f'pos_z_array')
    except:
        pass

    # pcm = ax.pcolormesh(xedges, yedges, kde, cmap=plt.cm.gist_earth_r,vmin=0,vmax=8)
    # fig.colorbar(pcm, ax=ax)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.savefig(f'{folder_name}_rg_rz.pdf',dpi=300)
    # plt.close()
    # print('end plot')
    # print('start plot')

    # x, kde_v = mf.kde_1d(pos_z_array, num_points=2000, xmin=0,xmax=20,bw_method=0.1)
    # fig, ax = plt.subplots()

    # pcm = ax.plot(x,kde_v,'k-')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.savefig(f'{folder_name}_pos_z_{0.1}.pdf',dpi=300)
    # plt.close()
    # print('end plot')



    end=datetime.datetime.now()
    print(f'start at {start}')
    print(f'end   at {end}')
    print(f'total wall time is  {end-start}')