'''
Author: qyp422
Date: 2022-10-17 15:29:44
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-02-26 21:29:45

Description: 

Copyright (c) 2022 by qyp422, All Rights Reserved. 
'''

import os,sys,datetime,time
import base

print('\n' + __file__ + ' is called\n')

class Lammps_dumpreader():
    def __init__(self, configuration,parameters,check_overlap=False):
        self.frame_num = 0
        self._conf = False
        self.total_atoms = False
        self.box_arr = False
        self._check_overlap = check_overlap
        if not os.path.isfile(configuration):
            sys.exit(f'cannot find configuration file {configuration}')
        if not os.path.isfile(parameters):
            sys.exit(f'cannot find parameters file {configuration}')
        # deal system topology
        
        self.para_dict = self.import_base_parameters(parameters)
        # set n_total
        if 'total_atoms' in self.para_dict:
            self.total_atoms = int(self.para_dict['total_atoms'])
        # set box_arr
        if 'box_arr' in self.para_dict:
            self.box_arr = [float(x) for x in self.para_dict['box_arr']]
        #get system
        self._system = self.get_system()
        if 'probe_mol' in self.para_dict:
            self._system.probe_mol = int(self.para_dict['probe_mol'])
            self._system.target_mol =  self._system._N_strands - self._system.probe_mol
        else:
            print('probe_mol not set')
        if 'probe_n' in self.para_dict:
            tem = int(self.para_dict['probe_n'])
            self._system.probe_n = tem
            self._system.probe = list(range(tem))
            self._system.target = list(range(tem,self.total_atoms))
            self._system.target_n= self._system._N - self._system.probe_n
        else:
            print('probe_n not set')
        
        self._conf = open(configuration, "r")

        if self._check_system_parameter():
            print('Lammps_dumpreader init done!!!!!\n')
            self._conf.close() 
            self._conf = open(configuration, "r")
        else:
            self._conf.close() 
            exit('topfile not fit with dumpfile!')

    def __del__(self):
        if self._conf:self._conf.close()

    def get_system(self):
        cm = [0,0,0]
        quat = [0,0,0,0]
        system = base.System(self.box_arr, time=-1)
        base.Nucleotide.index = 0
        base.Strand.index = 0
        s = base.Strand()
        strandid_current = 0
        if_circle = True
        for i in range(self.total_atoms):
            temlist = self.para_dict[str(i)]
            strandid = int(temlist[0])
            basenum = int(temlist[1])
            n5 = int(temlist[2])
            if n5 == -1: 
                if_circle = False
            if strandid != strandid_current:
                s._circular = if_circle
                system.add_strand(s,self._check_overlap)
                s = base.Strand()
                strandid_current = s.index
                if_circle = True
            s.add_nucleotide(base.Nucleotide(cm,quat,basenum,n5=n5))
        #add last strand
        s._circular = if_circle
        system.add_strand(s,self._check_overlap)
        
        #do some prepare for system 
        system.update_nucleotide_list()
        system.map_nucleotides_to_strands()
        print('get system ready!!')
        return system
    
    def _read_single_frame(self, only_strand_ends=False, skip=False):
        line = self._conf.readline()
        if  line == '':
            print('\nend of data reading!\n')
            return False
        if skip:
            # need to skip elf.total_atoms+8 lines
            self._system._time = int(self._conf.readline())
            for _ in range(self.total_atoms+7):
                self._conf.readline()
            print(f'data skip at system_time {self._system._time},total frame is {self.frame_num}!')
            self.frame_num += 1
            return True
        #   ITEM: TIMESTEP
        self._system._time = int(self._conf.readline())
        #   ITEM: NUMBER OF ATOMS
        self._conf.readline()
        # total_atoms
        self._conf.readline()
        #   ITEM: box BOUNDS pp pp pp
        self._conf.readline()
        # box xyz
        self._conf.readline()
        self._conf.readline()
        self._conf.readline()

        
        #   ITEM: ATOMS id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]
        self._conf.readline()
        for i in range(self.total_atoms):
            ls = self._conf.readline().split()
            cm = [float(x) for x in ls[3:6]]
            quat = [float(x) for x in ls[6:10]]
            self._system.update_nucleotide_pos_quat(i,cm=cm,quat=quat)
        
        print(f'data updata at system_time {self._system._time},total frame is {self.frame_num}!')
        self.frame_num += 1
        # print(self._system)
        return True

    def _check_system_parameter(self):
        line = self._conf.readline()
        if  line == '':
            print('\nno data!\n')
            return False
        #   ITEM: TIMESTEP
        self._conf.readline()
        #   ITEM: NUMBER OF ATOMS
        self._conf.readline()
        if self.total_atoms:
            if self.total_atoms != int(self._conf.readline()):
                print('\nwrong top file with total_atoms!\n')
                return False
        else:
            self.total_atoms = int(self._conf.readline())
        #   ITEM: box BOUNDS pp pp pp
        line = self._conf.readline()
        if self.box_arr:
            tem_box  =  [float(x) for x in self._conf.readline().split()]
            tem_box += [float(x) for x in self._conf.readline().split()]
            tem_box  += [float(x) for x in self._conf.readline().split()]
            for i,j in zip(self.box_arr,tem_box):
                if abs(i-j) > 0.0001:
                    print('\nwrong top file with boxsize!\n')
                    return False
        else:
            self.box_arr  =  [float(x) for x in self._conf.readline().split()] #x
            self.box_arr  += [float(x) for x in self._conf.readline().split()] #y   
            self.box_arr  += [float(x) for x in self._conf.readline().split()] #z
        return True

    @classmethod
    def import_base_parameters(self,filename):
        parameter_dict = {}
        try:
            r = open(filename)
        except:
            sys.exit(f'cannot find {filename} file')
        for line in r.readlines():
            if line.find('#') == -1:
                linelist = line.strip().split()
                if len(linelist)>2:
                    parameter_dict[linelist[0]] = linelist[1:]
                else:
                    parameter_dict[linelist[0]] = linelist[1]
        r.close()
        return parameter_dict

    def output_timestep_datafile(self,frame_num,path=os.getcwd(),output_filename = 'out.data',top_filename='top.data',deltabp =0):
        if deltabp :
            top_filename = 'topnew.data'

        if frame_num == 0:
            self._read_single_frame(skip = False)
            if deltabp :
                s = self._system.get_circle_rotation(bp_first=deltabp)
                s.update_probe_target_system(probe_mol= 1)
                top_message = s.get_tw_wr_lk(s._strands[0],s._strands[1])
                print('topmessage is ',top_message)
                s.get_lammps_data_file(output_filename, top_filename,path = path,message=f'nick at {deltabp} bp on second strand!')
                path_ovt = os.path.join(path,'ovt.data')
                with open(path_ovt,'w+') as f:
                    f.write(s.lammps_output)
                return True
            top_message = self._system.get_tw_wr_lk(self._system._strands[0],self._system._strands[1])
            print('topmessage is ',top_message)
            self._system.get_lammps_data_file(output_filename, top_filename,path = path)
            path_ovt = os.path.join(path,'ovt.data')
            with open(path_ovt,'w+') as f:
                f.write(self._system.lammps_output)
            return True
        while self._read_single_frame(skip = True):
            if self.frame_num == frame_num:
                self._read_single_frame(skip = False)
                if deltabp :
                    s = self._system.get_circle_rotation(bp_first=deltabp)
                    s.update_probe_target_system(probe_mol= 1)
                    top_message = s.get_tw_wr_lk(s._strands[0],s._strands[1])
                    print('topmessage is ',top_message)
                    s.get_lammps_data_file(output_filename, top_filename,path = path,message=f'nick at {deltabp} bp on second strand!')
                    path_ovt = os.path.join(path,'ovt.data')
                    with open(path_ovt,'w+') as f:
                        f.write(s.lammps_output)
                    return True
                top_message = self._system.get_tw_wr_lk(self._system._strands[0],self._system._strands[1])
                print('topmessage is ',top_message)
                self._system.get_lammps_data_file(output_filename, top_filename,path = path)
                path_ovt = os.path.join(path,'ovt.data')
                with open(path_ovt,'w+') as f:
                    f.write(self._system.lammps_output)
                return True
        print(f'cannot found {frame_num} th frame! ')
        return False


    def get_frame_system(self,frame_num):
        if frame_num == 0:
            self._read_single_frame(skip = False)
            return self._system
        while self._read_single_frame(skip = True):
            if self.frame_num == frame_num:
                self._read_single_frame(skip = False)
                return self._system
        print(f'cannot found {frame_num} th frame! ')
        return False
# cm = [] center of mass of each starnd no pre!
# rg = [] rg of mass of each starnd no pre!
# sor = [] sor for each stand
# hb[0]  = [hbbond , probe_mol ,target_mol, single_probe_mol , single_target_mol] 
# hb[1]  = every probe
# hb[2]  = number of target on a single probe

# lammps_output is the cm of nt output
# lammps_ovt_output is the base and hb cm of nt output
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
    r = Lammps_dumpreader(filename,topfile)
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
            

            #hb

            # r._system.get_hb_pair(mode = 'search')
            # # w.add_hb_message(t,r._system.hb_percentage)

            # w.add_nickheat(t,r._system.single())

            #lk

            # w.add_lk_message(t,r._system.get_tw_wr_lk(r._system._strands[0],r._system._strands[1]))

            #local twist


            w.add_local_twist(t,r._system.get_local_twist(r._system._strands[0],r._system._strands[1]))
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

    # r._system.get_system_graph(path=pwd,filename = filename)

    # w.add_kde_message(t,r._system.get_i_density_distribution(i = 2))
    


    del w
    del r
    end=datetime.datetime.now()
    print(f'start at {start}')
    print(f'end   at {end}')
    print(f'total wall time is  {end-start}')
    

    # plot 

    
if __name__ == "__main__":
        main()
