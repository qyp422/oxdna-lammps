'''
Author: qyp422
Date: 2023-03-14 15:24:46
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-03-14 15:37:44
Description: 

Copyright (c) 2023 by qyp422, All Rights Reserved. 
'''
import os,sys,datetime,time
import base
import mathfuction as mf
import dumpreader as dr
def main():
    start=datetime.datetime.now()
    
    if len(sys.argv)<3:
        sys.exit('Syntax: $> python dump_reader.py input_filename top_filename lk(option)')
    pwd = os.getcwd() 
    filename = os.path.basename(sys.argv[1])
    topfile = os.path.basename(sys.argv[2])
    try:
        lk_n = int(sys.argv[3])
    except:
        lk_n = 32

    # open file
    r = dr.Lammps_dumpreader(filename,topfile)
    w = base.File_system(pwd,filename)
    w._lk = lk_n
    w.n_probe = r._system.probe_mol
    w.n_target = r._system.target_mol

    frequency = 4000 # every times of f will output

    condition = True #frame 0 always read
    cpuflag = False # to compute cpu time
    while r._read_single_frame(skip = not condition):
        
        
        if condition:
            if cpuflag: start_d = time.perf_counter()
            
            t = r._system._time

            #sor

            # w.add_sor_message(t,r._system.sor)


            #rg no pre

            #w.add_rg_message(t,r._system.rgrsq)
            
            #math
            # data = r._system.get_contactmap(r._system._strands[0],r._system._strands[1])
            # ww.write(mf.get_supercoiling_shape(data))

            #hb

            r._system.get_hb_pair(mode = 'search')
            w.add_hb_message(t,r._system.hb_percentage)

            #hbgraph must get_hb_pair first
            # if (r.frame_num-1) % frequency == 0 :
            # #    r._system.get_system_graph()
            #     pass
            
            # z density_distribution
            
            # r._system.get_i_density_distribution(i = 2)
            
            #ovt
            # w.add_ovt_output(r._system.ovt_lammps_output)
        
            #cpu
            if cpuflag:
                end_d = time.perf_counter()
                print(f'this time {r.frame_num - 1} use cpu time {end_d - start_d:.3f}')
                w.cpu_time(t,end_d - start_d)

        # reader condition
        condition = True # if r.frame_num % frequency == 0 else False

    # close file

    r._system.get_system_graph(path=pwd,filename = filename)

    w.add_kde_message(t,r._system.get_i_density_distribution(i = 2))
    


    del w
    del r
    end=datetime.datetime.now()
    print(f'start at {start}')
    print(f'end   at {end}')
    print(f'total wall time is  {end-start}')
    

    # plot 

    
if __name__ == "__main__":
        main()