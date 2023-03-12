'''
Author: qyp422
Date: 2022-11-12 11:09:37
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-03-06 20:40:46
Description: 

Copyright (c) 2022 by qyp422, All Rights Reserved. 
'''

import base
import numpy as np
import math
import sys,os
import mathfuction as mf
import random

BP = "bp"
DEGREES = "degrees"
DEG = "degrees"
RAD = "radiants"
RADIANTS = "radiants"
BASE_PAIRS = "bp"

print('\n' + __file__ + ' is called\n')

class StrandGenerator():

    @classmethod
    def generate(self, bp, sequence=None, start_pos=np.array([0, 0, 0]), dir=np.array([0, 0, 1]), perp=np.array([1, 0, 0]), rot=0., double=True, circular=False, DELTA_LK=0, BP_PER_TURN=10.5, ds_start=None, ds_end=None, force_helicity=False):
        """
        Generate a strand of DNA.
            - linear, circular (circular)
            - ssDNA, dsDNA (double)
            - Combination of ss/dsDNA (ds_start, ds_end)
            Note: Relevent argument(s) in parentheses.

        Arguments:
        bp --- Integer number of bp/nt (required)
        sequence --- Array of integers or string. Should be same length as bp (default None)
            Default (None) generates a random sequence.
            Ex: [0,1,2,3,0]
            Ex: "AGCTA"
            See dictionary base.base_to_number for int/char conversion {0:'A'}
        start_pos --- Location to begin building the strand (default np.array([0, 0, 0]))
        dir --- a3 vector, orientation of the base (default np.array([0, 0, 1]))
        perp --- Sets a1 vector, the orientation of the backbone. (default False)
            Must be perpendicular to dir (as a1 must be perpendicular to a3)
            If perp is None or False, perp is set to a random orthogonal angle
        rot --- Rotation of first bp (default 0.)
        double --- Generate dsDNA (default True)
        circular --- Generate closed circular DNA (defalt False)
            Limitations...
            For ssDNA (double=False): bp >= 4
            For dsDNA (double=True) : bp >= 30
            Will throw warnings. Allowed, but use at your own risk.
        DELTA_LK --- Integer change in linking number from Lk0 (default 0)
            Only valid if circular==True
        BP_PER_TURN --- Base pairs per complete 2*pi helix turn. (default 10.34)
            Only valid if circular==True
        ds_start --- Index (from 0) to begin double stranded region (default None)
        ds_end --- Index (from 0) to end double stranded region (default None)
            Default is None, which is entirely dsDNA; sets ds_start = 0, ds_end=bp
            Ex: ds_start=0, ds_end=10 will create a double stranded region on bases
                range(0,10): [0,1,2,3,4,5,6,7,8,9]
            Note: To generate a nicked circular dsDNA, manually change state with
                  {Strand}.make_noncircular()
        force_helicity --- Force generation of helical strands. Use helicity by default
            for bp > 30. Warns from 18 to 29. Will crash oxDNA below 18. (default False)

        Note: Minimuim circular duplex is 18. Shorter circular strands disobey FENE.
        For shorter strands, circular ssDNA is generated in a circle instead of having
        imposed helicity.

        Examples:
        Generate ssDNA:
            generate(bp=4,sequence=[0,1,2,3],double=False,circular=False)
        Generate circular dsDNA with +2 Linking number:
            generate(bp=45,double=True,circular=True,DELTA_LK=2)
        Generate a circular ssDNA (45nt) with ssDNA (25nt) annealed to indices 0 to 24:
            generate(bp=45,double=True,circular=True,ds_start=0,ds_end=25)
        """
        # we need numpy array for these
        start_pos = np.array(start_pos, dtype=float)
        dir = np.array(dir, dtype=float)
        if isinstance(sequence, list):
            sequence = np.array(sequence)

        # Loads of input checking...
        if isinstance(sequence, str):
            try:
                sequence = np.array([base.base_to_number[x] for x in sequence])
            except:
                sys.exit("Key Error: sequence is invalid")
        if any(sequence == None):
            sequence = np.random.randint(0, 4, bp)
        elif len(sequence) != bp:
            n = bp - len(sequence)
            sequence = np.append(sequence, np.random.randint(0, 4, n))
            print("sequence is too short, adding %d random bases" % n)

        if circular == True and bp < 30:
            # 30 is about the cut off for circular dsDNA. Anything shorter will probably clash.
            # oxDNA can relax down to 18.
            # 4 is about the cut off for circular ssDNA. Use dsDNA cutoff for saftey.
            print("sequence is too short! Proceed at your own risk")

        option_use_helicity = True
        # if circular == True and bp < 30 and double == False:
        #     base.Logger.log("sequence is too short! Generating ssDNA without imposed helicity", base.Logger.WARNING)
        #     # Do not impose helcity to generate shorter circular ssDNA
        #     if not force_helicity:
        #         option_use_helicity = False

        if ds_start == None:
            ds_start = 0
        if ds_end == None:
            ds_end = bp
        if ds_start > ds_end:
            sys.exit("ds_end > ds_start")
        if  ds_end > bp:
            sys.exit("ds_end > bp")

        # we need to find a vector orthogonal to dir
        dir_norm = np.sqrt(np.dot(dir,dir))
        if dir_norm < 1e-10:
            print("direction must be a valid vector, defaulting to (0, 0, 1)")
            dir = np.array([0, 0, 1])
        else:
            dir /= dir_norm

        if perp is None or perp is False:
            v1 = np.random.random_sample(3)
            v1 -= dir * (np.dot(dir, v1))
            v1 /= np.sqrt(sum(v1*v1))
        else:
            v1 = perp

        # Setup initial parameters
        ns1 = base.Strand()
        # and we need to generate a rotational matrix
        R0 = mf.get_rotation_matrix(dir, rot)
        #R = get_rotation_matrix(dir, np.deg2rad(35.9))
        R = mf.get_rotation_matrix(dir, [1, BP])
        a1 = v1
        a1 = np.dot (R0, a1)
        rb = np.array(start_pos)
        a3 = dir

        # Circular strands require a continuious deformation of the ideal helical pitch
        if circular == True:
            # Unit vector orthogonal to plane of torus
            # Note: Plane of torus defined by v1,dir
            torus_perp = np.cross(v1,dir)
            # bp-bp rotation factor to yield a smooth deformation along the torus
            smooth_factor = np.mod(bp,BP_PER_TURN)/float(bp) + 1
            # Angle between base pairs along torus
            angle = 2. * np.pi / float(bp)
            # Radius of torus
            radius = base.FENE_R0_OXDNA2 / math.sqrt(2. * (1. - math.cos(angle)))

        if circular == True and option_use_helicity:
            # Draw backbone in a helical spiral around a torus
            # Draw bases pointing to center of torus
            for i in range(bp):
                # Torus plane defined by dir and v1
                v_torus = v1 *base.BASE_BASE * math.cos(i * angle) + \
                        dir * base.BASE_BASE * math.sin(i * angle)
                rb += v_torus

                # a3 is tangent to the torus
                a3 = v_torus/np.linalg.norm(v_torus)
                R = mf.get_rotation_matrix(a3, [i * (round(bp/BP_PER_TURN) + DELTA_LK)/float(bp) * 360, DEGREES])

                # a1 is orthogonal to a3 and the torus normal
                a1 = np.cross (a3, torus_perp)

                # Apply the rotation matrix
                a1 = np.dot(R, a1)
                ns1.add_nucleotide(base.Nucleotide(rb - base.CM_CENTER_DS * a1, mf.exyz_to_quat(a1,a3) , sequence[i]))
            ns1.make_circular(check_join_len=True)
        elif circular == True and not option_use_helicity:
            for i in range(bp):
                rbx = math.cos (i * angle) * radius + 0.34 * math.cos(i * angle)
                rby = math.sin (i * angle) * radius + 0.34 * math.sin(i * angle)
                rbz = 0.
                rb = np.array([rbx, rby, rbz])
                a1x = math.cos (i * angle)
                a1y = math.sin (i * angle)
                a1z = 0.
                a1 = np.array([a1x, a1y, a1z])
                ns1.add_nucleotide(base.Nucleotide(rb, mf.exyz_to_quat(a1, np.array([0, 0, 1])), sequence[i]))
            ns1.make_circular(check_join_len=True)
        else:
            # Add nt in canonical double helix
            for i in range(bp):
                ns1.add_nucleotide(base.Nucleotide(rb - base.CM_CENTER_DS * a1, mf.exyz_to_quat(a1, a3), sequence[i]))
                if i != bp-1:
                    a1 = np.dot(R, a1)
                    rb += a3 * base.BASE_BASE

        # Fill in complement strand
        if double == True:
            ns2 = base.Strand()
            for i in reversed(range(ds_start, ds_end)):
                # Note that the complement strand is built in reverse order
                nt = ns1._nucleotides[i]
                a1 = -nt._a1
                a3 = -nt._a3
                nt2_cm_pos = -(base.FENE_EPS + 2*base.POS_BACK) * a1 + nt.cm_pos
                ns2.add_nucleotide(base.Nucleotide(nt2_cm_pos, mf.exyz_to_quat(a1, a3), 3-sequence[i]))
            if ds_start == 0 and ds_end == bp and circular == True:
                ns2.make_circular(check_join_len=True)
            return ns1, ns2
        else:
            return ns1
    '''
    description: 
    param {*} v1_l distance of first end last
    param {*} degree rotation of fisrt and last
    return {*} Strand class
    '''
    @classmethod
    def generate_circle_fixedends(self, bp, sequence=None, start_pos=np.array([0, 0, 0]), dir=np.array([0, 0, 1]), perp=np.array([1, 0, 0]), rot=0., double=True, circular=False, DELTA_LK=0, BP_PER_TURN=10.5, ds_start=None, ds_end=None,v1_l = 1,degree = 0):
        # we need numpy array for these
        start_pos = np.array(start_pos, dtype=float)
        dir = np.array(dir, dtype=float)
        if isinstance(sequence, list):
            sequence = np.array(sequence)

        # Loads of input checking...
        if isinstance(sequence, str):
            try:
                sequence = np.array([base.base_to_number[x] for x in sequence])
            except:
                sys.exit("Key Error: sequence is invalid")
        if any(sequence == None):
            sequence = np.random.randint(0, 4, bp)
        elif len(sequence) != bp:
            n = bp - len(sequence)
            sequence = np.append(sequence, np.random.randint(0, 4, n))
            print("sequence is too short, adding %d random bases" % n)

        if circular == True and bp < 30:
            # 30 is about the cut off for circular dsDNA. Anything shorter will probably clash.
            # oxDNA can relax down to 18.
            # 4 is about the cut off for circular ssDNA. Use dsDNA cutoff for saftey.
            print("sequence is too short! Proceed at your own risk")

        option_use_helicity = True
        # if circular == True and bp < 30 and double == False:
        #     base.Logger.log("sequence is too short! Generating ssDNA without imposed helicity", base.Logger.WARNING)
        #     # Do not impose helcity to generate shorter circular ssDNA
        #     if not force_helicity:
        #         option_use_helicity = False

        if ds_start == None:
            ds_start = 0
        if ds_end == None:
            ds_end = bp
        if ds_start > ds_end:
            sys.exit("ds_end > ds_start")
        if  ds_end > bp:
            sys.exit("ds_end > bp")

        # we need to find a vector orthogonal to dir
        dir_norm = np.sqrt(np.dot(dir,dir))
        if dir_norm < 1e-10:
            print("direction must be a valid vector, defaulting to (0, 0, 1)")
            dir = np.array([0, 0, 1])
        else:
            dir /= dir_norm

        if perp is None or perp is False:
            v1 = np.random.random_sample(3)
            v1 -= dir * (np.dot(dir, v1))
            v1 /= np.sqrt(sum(v1*v1))
        else:
            v1 = perp

        # Setup initial parameters
        ns1 = base.Strand()
        # and we need to generate a rotational matrix
        R0 = mf.get_rotation_matrix(dir, rot)
        #R = get_rotation_matrix(dir, np.deg2rad(35.9))
        R = mf.get_rotation_matrix(dir, [(round(bp/BP_PER_TURN) + DELTA_LK)/float(bp-1) * 360, DEGREES])
        a1 = v1
        a1 = np.dot (R0, a1)
        v1_l = v1_l
        rb = np.array(start_pos)
        a3 = dir
        rb += (v1_l/2 + v1_l/(bp-1) - base.BASE_BASE)*v1
        
        # Circular strands require a continuious deformation of the ideal helical pitch
        if circular == True:
            # Unit vector orthogonal to plane of torus
            # Note: Plane of torus defined by v1,dir
            torus_perp = np.cross(v1,dir)
            # bp-bp rotation factor to yield a smooth deformation along the torus
            smooth_factor = np.mod(bp,BP_PER_TURN)/float(bp) + 1
            # Angle between base pairs along torus
            angle =  2*np.pi  / (bp-1)
            # Radius of torus
            radius = base.FENE_R0_OXDNA2 / math.sqrt(2. * (1. - math.cos(angle)))
        
        if circular == True and option_use_helicity:
            # Draw backbone in a helical spiral around a torus
            # Draw bases pointing to center of torus
            for i in range(bp):
                # Torus plane defined by dir and v1
                v_torus = v1 *base.BASE_BASE * np.cos((i) * angle ) + \
                        dir * base.BASE_BASE * np.sin((i) * angle ) - (v1 *v1_l)/(bp-1)

                rb += v_torus
                
                # a3 is tangent to the torus
                a3 = v_torus/np.linalg.norm(v_torus)
                R = mf.get_rotation_matrix(a3, [i * (round(bp/BP_PER_TURN) + DELTA_LK)/float(bp-1) * 360, DEGREES])

                # a1 is orthogonal to a3 and the torus normal
                a1 = np.cross (a3, torus_perp)
                
                # Apply the rotation matrix
                a1 = np.dot(R, a1)
                # print(a1)
                ns1.add_nucleotide(base.Nucleotide(rb - base.CM_CENTER_DS * a1, mf.exyz_to_quat(a1,a3) , sequence[i]))
            ns1.make_circular(check_join_len=True)
        # Fill in complement strand
        else:
            # Add nt in canonical double helix
            for i in range(bp):
                ns1.add_nucleotide(base.Nucleotide(rb - base.CM_CENTER_DS * a1, mf.exyz_to_quat(a1, a3), sequence[i]))
                if i != bp-1:
                    a1 = np.dot(R, a1)
                    rb += a3 * base.BASE_BASE
        if double == True:
            ns2 = base.Strand()
            for i in reversed(range(ds_start, ds_end)):
                # Note that the complement strand is built in reverse order
                nt = ns1._nucleotides[i]
                a1 = -nt._a1
                a3 = -nt._a3
                nt2_cm_pos = -(base.FENE_EPS + 2*base.POS_BACK) * a1 + nt.cm_pos
                ns2.add_nucleotide(base.Nucleotide(nt2_cm_pos, mf.exyz_to_quat(a1, a3), 3-sequence[i]))
            if ds_start == 0 and ds_end == bp and circular == True:
                ns2.make_circular(check_join_len=True)
            return ns1, ns2
        else:
            return ns1

    @classmethod
    def gen_probe_target(self,box,probe_array,z_tem_wall,n_target=0,seq='CC',neighbor=0):
        #target seq
        seqt = np.array([7-base.base_to_number[x] for x in seq])
        #probe seq
        seqp = seq
        s = base.System(np.array(box))
        if isinstance(probe_array,int):
            probe_array = [probe_array,probe_array]

        # set Initial 
        boxsize = s.l_box[:2]
        bool_double = False
        bool_circular = False
        shift = boxsize/probe_array/2

        pos_range_x = np.linspace(s._box[0]+shift[0],s._box[1]-shift[0],probe_array[0],endpoint=True)
        pos_range_y = np.linspace(s._box[2]+shift[1],s._box[3]-shift[1],probe_array[1],endpoint=True)
        # crate target
        if neighbor == 0:
            for i in pos_range_x:
                for j in pos_range_y:
                    s.add_strands(self.generate(len(seqp), sequence=seqp,start_pos=np.array([i,j,0]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)
        # elif neighbor == 2:
        #     for i in range(n_probe):
        #         for j in range(n_probe):
        #             if i % 2 == 0:
        #                 s.add_strands(g.generate(len(seq), sequence=seq,start_pos=np.array([pos_range[i],pos_range[j],0]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)
        #             else:
        #                 s.add_strands(g.generate(len(seq), sequence=seq,start_pos=np.array([pos_range[i],pos_range[j],0]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)
        #             print(pos_range[i],pos_range[j])
        # elif neighbor == 4:
        #     for i in range(n_probe):
        #         for j in range(n_probe):
        #             if (i+j) % 2 == 0:
        #                 s.add_strands(g.generate(len(seq), sequence=seq,start_pos=np.array([pos_range[i],pos_range[j],0]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)
        #             else:
        #                 s.add_strands(g.generate(len(seq), sequence=seq,start_pos=np.array([pos_range[i],pos_range[j],0]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)
        #             print(pos_range[i],pos_range[j])


        # s.add_strands(g.generate_complementary_strand(s._strands[1]), check_overlap=False)

        #create target
        
        if n_target > boxsize[0]*boxsize[0]:
            print('number error')
        
        l = int(np.sqrt(n_target-1))+1 # lenth of targets
        shift = boxsize/l/2
        pos_range = np.linspace(s._box[0]+shift[0],s._box[1]-shift[0],l,endpoint=True)
        for i in range(n_target):
            s.add_strands(self.generate(len(seqt), sequence=seqt,start_pos=np.array([pos_range[i//l],pos_range[i%l],z_tem_wall]), double=bool_double, circular=bool_circular, DELTA_LK=0), check_overlap=False)

        s.update_probe_target_system(probe_mol=probe_array[0]*probe_array[1])
        return s
        # s.get_lammps_data_file("probe8type.data", "top.data")
        # with open('probe8type_ovt.data','w+') as f:
        #     f.write(s.lammps_output)
        # print('-----lammps generated done!')

class Lammps_in_file():
    @classmethod
    def in_file_generate_with_relax_8type_fixwall(self,in_file_name,read_file_name,dimension = 3, num_file = 1,T = 300,seq = 'seqdep',salt = 0.2,damp = 2.5,timestep = 0.005,time = 100000000,n_probe=36,n_target=1,bplen=30,surface_density=-1,target_density=-1,runflag=True,path=os.getcwd()):
        # all the probe is type  
        t_relaxation = 10**5 
        t_r = 10**7 # for oxdna relaxation of inital stucture t should be o(10^6) 
        k = n_probe # another symbol n_target
        wall_molecule = n_target+n_probe+1
        temperature = T/3000.0
        output_freq = 100000
        energy_freq = 100000
        in_name = str(in_file_name) + '_T_' + str(T)
        Ax = '0.0'
        Ay = '0.0'
        Az = '0.0'
        Vx = '0.0'
        Vy = '0.0'
        Vz = '0.0'
        dnanve = ''
        fixdna = ''
        damp_relax = 78.9375
        # optional parameters
        only_dna_model = True #True for only output DNA lammpstrj
        neighbor_style = 'nsq' #bin or nsq for neighbor
        neighbor_style_dict = {'bin':'2.0 bin','nsq':'1.0 nsq'} #dict of neighbor
    
        balance_thresh = 1.05 # imbalance threshold that must be exceeded to perform a re-balance
        balance_style = 'rcb' # rcb or shift
        balance_style_dict = {'rcb':'rcb','shift':'shift'}
        # check style
        if neighbor_style not in neighbor_style_dict:
            sys.exit('neighbor_style wrong')
        if balance_style not in balance_style_dict:
            sys.exit('balance_style wrong')
    
        #balance_fix input script
        if balance_style == 'rcb':
            balance_fix = 'comm_style tiled\n'
            balance_fix += 'fix 3 all balance 1000 ' + str(balance_thresh) + ' ' + balance_style_dict[balance_style] + '\n'
        if balance_style == 'shift':
            balance_fix = 'fix 3 all balance 1000 ' + str(balance_thresh) + ' ' + balance_style_dict[balance_style] + ' xyz 10 ' + str(balance_thresh) +'\n'
    
        # # only 1 61 ... are fix
        # for i in range(k):
        #     dnanve += str(2+i*60) + ':' + str(60+i*60) + ' '
        # fixdna += '1:' + str(k*60-59) + ':60'
        for i in range(k):
            # fix every probe first and fifth nt
            dnanve += str(2+i*bplen) + ':' + str(bplen+i*bplen) + ' '
            fixdna += str(1+i*bplen) + ' '
        # add target id
        dnanve += str(k*bplen+1) + ':' + str(k*bplen+n_target*bplen)
    
        # system
        ll = '# lammps in file with probe molecule %d with type 1 2 3 4\n'%(k)
        ll+= '# lammps in file with target molecule %d with type 5 6 7 8\n'%(n_target)
        ll+= '# DNA bp_len is %d\n'%(bplen)
        ll+= '# LAMMPS data file probe surface density is %f\n' % surface_density
        ll+= '# LAMMPS data file target density is %f\n' % target_density
        ll+= '# probe fix by only fisrt bp\n'
    
        for i in range(num_file):
            random_seed = random.randint(1, 50000) #set random for langevin
            outputfile_name = 'original_T_' +  str(T) + '_' + str(i) + '.lammpstrj' #set outputfile_name
            prefile_name = 'pre_T_' +  str(T) + '_' + str(i) + '.lammpstrj' #set prefile_name
            outputenergy_name = 'original_T_' +  str(T) + '_' + str(i) +  '_energy.txt'
            preenergy_name = 'pre_T_' +  str(T) + '_' + str(i) +  '_energy.txt'
            temperature_l = temperature #temperature for langevin
            out = open(os.path.join(path,str(in_name)+ '_' + str(i)),'w')
            out.write(ll)
            out.write('variable ofreq	equal %d\n' % output_freq)
            out.write('variable efreq	equal %d\n' % energy_freq)
            out.write('echo	screen\n')
            out.write('units lj\n')
            out.write('\n')
            out.write('dimension %d\n'% dimension)
            out.write('\n')
            out.write('newton on\n')
            out.write('\n')
            out.write('boundary  p p p\n')
            out.write('atom_style hybrid bond ellipsoid oxdna\n')
            out.write('atom_modify sort 0 1.0\n')
            out.write('\n')
            out.write('# Pair interactions require lists of neighbours to be calculated\n')
            out.write('neighbor %s\n'% neighbor_style_dict[neighbor_style])
            out.write('neigh_modify every 1 delay 0 check yes\n')
            out.write('\n')
            out.write('read_data %s\n'% read_file_name)
            out.write('set atom * mass 3.1575\n')
            out.write('\n')
            out.write('group all type 1 2 3 4 5 6 7 8\n')
            out.write('group dna type 1 2 3 4 5 6 7 8\n')
            out.write('group dnanve id %s\n'% dnanve)
            out.write('group fixdna id %s\n'% fixdna)
            out.write('\n')
            out.write('# oxDNA bond interactions - FENE backbone\n')
            out.write('bond_style oxdna2/fene\n')
            out.write('bond_coeff * 2.0 0.25 0.7564\n')
            out.write('special_bonds lj 0 1 1\n')
            out.write('\n')
            out.write('# oxDNA pair interactions\n')
            out.write('pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh\n')
            out.write('pair_coeff *8 *8 oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32\n')
            out.write('pair_coeff *8 *8 oxdna2/stk     %s %.4f 1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65\n'% (seq,temperature))
            out.write('pair_coeff *8 *8 oxdna2/hbond   %s 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 1 4 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 2 3 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 5 8 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 6 7 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff *8 *8 oxdna2/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68\n')
            out.write('pair_coeff *8 *8 oxdna2/coaxstk 58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793\n')
            out.write('pair_coeff *8 *8 oxdna2/dh      %.4f %.4f 0.815\n'% (temperature,salt))
            out.write('\n')
            out.write('# NVE ensemble\n')
            out.write('fix 1 dnanve nve/dotc/langevin %.4f %.4f %.4f %d angmom 10\n'% (temperature_l,temperature_l,damp_relax,random_seed))
            out.write('\n')
            # out.write('comm_style tiled\n')
            # out.write('comm_modify cutoff 5.0\n') updata to balance_fix
            out.write(balance_fix)
            out.write('\n')
            out.write('# lj wall\n')
            out.write('fix 2 dnanve wall/lj126 zlo EDGE 1.0 1.0 1.12246 zhi EDGE 1.0 1.0 1.12246 pbc yes\n')
    
            out.write('# fixmove \n')
            out.write('variable	Ax atom %s \n'% Ax)
            out.write('variable	Ay atom %s \n'% Ay)
            out.write('variable	Az atom %s \n'% Az)
            out.write('variable	Vx atom %s \n'% Vx)
            out.write('variable	Vy atom %s \n'% Vy)
            out.write('variable	Vz atom %s \n'% Vz)
            out.write('fix 12 fixdna move variable v_Ax v_Ay v_Az v_Vx v_Vy v_Vz \n')
            out.write('\n')
            out.write('timestep %f\n'% timestep)
            out.write('\n')
            out.write('# output\n')
            if only_dna_model:
                out.write('compute quat dna property/atom quatw quati quatj quatk\n')
                out.write('dump quat dna custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , prefile_name))
            else:
                out.write('compute quat all property/atom quatw quati quatj quatk\n')
                out.write('dump quat all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , prefile_name))
            out.write('dump_modify quat sort id\n')
            out.write('compute erot dna erotate/asphere\n')
            out.write('compute ekin dna ke\n')
            out.write('compute peratom dna pe/atom\n')
            out.write('compute epot dna reduce sum c_peratom\n')
            out.write('variable erot equal c_erot\n')
            out.write('variable ekin equal c_ekin\n')
            out.write('variable epot equal c_epot\n')
            out.write('variable etot equal c_erot+c_ekin+c_epot\n')
            out.write('fix 5 all print ${efreq} "$(step)  ekin = ${ekin} |  erot = ${erot} | epot = ${epot} | etot = ${etot}" file %s\n' % preenergy_name )
            out.write('\n')
            out.write('# relaxation time\n')
            out.write('run %d\n'% t_r)
            out.write('run %d\n'% t_r)
            if runflag:
                out.write('\n')
                out.write('# give system hybrid\n')
    
                out.write('pair_coeff 1 8 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
                out.write('pair_coeff 2 7 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
                out.write('pair_coeff 4 5 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
                out.write('pair_coeff 3 6 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
    
                out.write('# undump\n')
                out.write('undump quat\n')
                out.write('unfix 5\n')
                out.write('unfix 1\n')        
                out.write('\n')
                out.write('\n')
                out.write('# MD simulation step time\n')
                out.write('fix 30 dnanve nve/dotc/langevin %.4f %.4f %.4f %d angmom 10\n'% (temperature_l,temperature_l,damp,random_seed))
                out.write('\n')  
                if only_dna_model:
                    out.write('dump quatt dna custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , outputfile_name))
                else:
                    out.write('dump quatt all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , outputfile_name))
                out.write('dump_modify quatt sort id\n')
                out.write('fix 6 all print ${efreq} "$(step)  ekin = ${ekin} |  erot = ${erot} | epot = ${epot} | etot = ${etot}" file %s\n' % outputenergy_name )
                out.write('\n')      
                out.write('reset_timestep 0\n')     
                out.write('run %d\n'% time)
                out.write('run %d\n'% time)
                out.write('run %d\n'% time)
                out.write('run %d\n'% time)
                out.write('\n')
                out.write('\n')
                out.write('\n')
    
            out.close()
            
    
        print('generate %d infile done' %num_file)

    @classmethod
    def in_file_generate_21(self,in_file_name,read_file_name,dimension = 3, num_file = 1,T = 300,seq = 'seqdep',salt = 0.2,damp = 1.0,timestep = 0.002,time = 100000000,path=os.getcwd(),outputfre = 100000):
        temperature = T/3000
        output_freq = outputfre
        energy_freq = outputfre
        in_name = str(in_file_name) + '_T_' + str(T)
        # system
        ll = '# lammps in file with supercoling with type 1 2 3 4 nve/dot\n'
        for i in range(num_file):
            random_seed = random.randint(1, 50000) #set random for langevin
            outputfile_name = 'original_T_' +  str(T) + '_' + str(i) + '.lammpstrj' #set outputfile_name
            outputenergy_name = outputfile_name + 'energy.txt'
            temperature_l = temperature #temperature for langevin
            out = open(os.path.join(path,str(in_name)+ '_' + str(i)),'w')
            out.write(ll)
            out.write('variable ofreq	equal %d\n' % output_freq)
            out.write('variable efreq	equal %d\n' % energy_freq)
            out.write('echo	screen\n')
            out.write('units lj\n')
            out.write('\n')
            out.write('dimension %d\n'% dimension)
            out.write('\n')
            out.write('newton on\n')
            out.write('\n')
            out.write('boundary  p p p\n')
            out.write('atom_style hybrid bond ellipsoid oxdna\n')
            out.write('atom_modify sort 0 1.0\n')
            out.write('\n')
            out.write('# Pair interactions require lists of neighbours to be calculated\n')
            out.write('neighbor 1.0 nsq\n')
            out.write('neigh_modify every 1 delay 0 check yes\n')
            out.write('\n')
            out.write('read_data %s\n'% read_file_name)
            out.write('set atom * mass 3.1575\n')
            out.write('\n')
            out.write('group all type 1 2 3 4\n')
            out.write('group dna type 1 2 3 4\n')
            out.write('\n')
            out.write('# oxDNA bond interactions - FENE backbone\n')
            out.write('bond_style oxdna2/fene\n')
            out.write('bond_coeff * 2.0 0.25 0.7564\n')
            out.write('special_bonds lj 0 1 1\n')
            out.write('\n')
            out.write('# oxDNA pair interactions\n')
            out.write('pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh\n')
            out.write('pair_coeff * * oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32\n')
            out.write('pair_coeff * * oxdna2/stk     %s %.4f 1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65\n'% (seq,temperature))
            out.write('pair_coeff * * oxdna2/hbond   %s 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 1 4 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 2 3 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff * * oxdna2/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68\n')
            out.write('pair_coeff * * oxdna2/coaxstk 58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793\n')
            out.write('pair_coeff * * oxdna2/dh      %.4f %.4f 0.815\n'% (temperature,salt))
            out.write('\n')
            out.write('# NVE ensemble\n')
            out.write('fix 1 all nve/dotc/langevin %.4f %.4f %.4f %d angmom 10\n'% (temperature_l,temperature_l,damp,random_seed))
            out.write('\n')
            out.write('comm_style tiled\n')
    #        out.write('comm_modify cutoff 2.5\n')
            out.write('fix 3 all balance 1000 1.05 rcb\n')
            out.write('\n')
            out.write('timestep %f\n'% timestep)
            out.write('\n')
            out.write('# output\n')
            out.write('compute quat all property/atom quatw quati quatj quatk\n')
            out.write('dump quat all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , outputfile_name))
            out.write('dump_modify quat sort id\n')
            out.write('compute erot dna erotate/asphere\n')
            out.write('compute ekin dna ke\n')
            out.write('compute peratom dna pe/atom\n')
            out.write('compute epot dna reduce sum c_peratom\n')
            out.write('variable erot equal c_erot\n')
            out.write('variable ekin equal c_ekin\n')
            out.write('variable epot equal c_epot\n')
            out.write('variable etot equal c_erot+c_ekin+c_epot\n')
            out.write('fix 5 all print ${efreq} "$(step)  ekin = ${ekin} |  erot = ${erot} | epot = ${epot} | etot = ${etot}" file %s\n' % outputenergy_name )
            out.write('\n')
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('\n')
            out.write('\n')
            out.write('\n')

            out.close()
        print(f'lammps infile done! total {num_file} files\n')

    @classmethod
    def in_file_generate_21_break_fixed_ends_plus_rotation(self,in_file_name,read_file_name,dimension = 3, num_file = 1,T = 300,seq = 'seqdep',salt = 0.2,damp = 2.5,timestep = 0.01,time = 100000000,path=os.getcwd(),outputfre = 100000,bp_num = 336,fixed_bp_len = 1,degree= 0,first_end_d = -1):
        temperature = T/3000
        output_freq = outputfre
        energy_freq = outputfre
        in_name = str(in_file_name) + '_T_' + str(T)
        #ends group set 
        Ax = '0.0'
        Ay = '0.0'
        Az = '0.0'
        Vx = '0.0'
        Vy = '0.0'
        Vz = '0.0'
        dnanve = ''
        fixdna = ''
        moveleft = ''
        moveright = ''
        dnanve += f'{fixed_bp_len+1}:{bp_num-fixed_bp_len} {bp_num+1+fixed_bp_len}:{2*bp_num-fixed_bp_len}'
        
        for i in range(fixed_bp_len):
            moveleft += f'{bp_num-i} {bp_num+i+1} '
            moveright += f'{i+1} {2*bp_num-i} '
        fixdna += moveleft + moveright
        # rotation set
        if degree:
            if first_end_d < 2:
                exit('you must know what u are doing!first and end bp is overlap!')
            posright = first_end_d/2.0
            posleft = - first_end_d/2.0
            v_rotation = 10000 #360 degree time
            time_rotate = int(degree/360*v_rotation/timestep)


        # system
        ll = '# lammps in file with supercoling with type 1 2 3 4 nve/dot\n'
        ll+= f'# total {bp_num} base pair\n'
        ll+= f'# fixed ends mode fixed {fixed_bp_len} bp of ends\n'
        ll+= f'# ends distance is  {first_end_d}\n'
        ll+= f'# fixed ends mode rotation {degree} o of ends\n'
        for i in range(num_file):
            random_seed = random.randint(1, 50000) #set random for langevin
            outputfile_name = 'original_T_' +  str(T) + '_' + str(i) + '.lammpstrj' #set outputfile_name
            outputenergy_name = outputfile_name + 'energy.txt'
            temperature_l = temperature #temperature for langevin
            out = open(os.path.join(path,str(in_name)+ '_' + str(i)),'w')
            out.write(ll)
            out.write('variable ofreq	equal %d\n' % output_freq)
            out.write('variable efreq	equal %d\n' % energy_freq)
            out.write('echo	screen\n')
            out.write('units lj\n')
            out.write('\n')
            out.write('dimension %d\n'% dimension)
            out.write('\n')
            out.write('newton on\n')
            out.write('\n')
            out.write('boundary  p p p\n')
            out.write('atom_style hybrid bond ellipsoid oxdna\n')
            out.write('atom_modify sort 0 1.0\n')
            out.write('\n')
            out.write('# Pair interactions require lists of neighbours to be calculated\n')
            out.write('neighbor 1.0 nsq\n')
            out.write('neigh_modify every 1 delay 0 check yes\n')
            out.write('\n')
            out.write('read_data %s\n'% read_file_name)
            out.write('set atom * mass 3.1575\n')
            out.write('\n')
            out.write('group all type 1 2 3 4\n')
            out.write('group dna type 1 2 3 4\n')
            out.write('group dnanve id %s\n'% dnanve)
            out.write('group fixdna id %s\n'% fixdna)
            out.write('group moveleft id %s\n'% moveleft)
            out.write('group moveright id %s\n'% moveright)
            out.write('\n')
            out.write('# oxDNA bond interactions - FENE backbone\n')
            out.write('bond_style oxdna2/fene\n')
            out.write('bond_coeff * 2.0 0.25 0.7564\n')
            out.write('special_bonds lj 0 1 1\n')
            out.write('\n')
            out.write('# oxDNA pair interactions\n')
            out.write('pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh\n')
            out.write('pair_coeff * * oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32\n')
            out.write('pair_coeff * * oxdna2/stk     %s %.4f 1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65\n'% (seq,temperature))
            out.write('pair_coeff * * oxdna2/hbond   %s 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 1 4 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 2 3 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff * * oxdna2/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68\n')
            out.write('pair_coeff * * oxdna2/coaxstk 58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793\n')
            out.write('pair_coeff * * oxdna2/dh      %.4f %.4f 0.815\n'% (temperature,salt))
            out.write('\n')
            out.write('# NVE ensemble\n')
            out.write('fix 1 dnanve nve/dotc/langevin %.4f %.4f %.4f %d angmom 10\n'% (temperature_l,temperature_l,damp,random_seed))
            out.write('\n')

            # rotation 
            if degree :
                out.write(f'fix 32 moveright  move rotate {posright:.4f} 0.0 0.0 0.0 -1.0 0.0 {v_rotation} \n')
                out.write(f'fix 33 moveleft move rotate {posleft:.4f}  0.0 0.0 0.0 1.0 0.0 {v_rotation} \n')
                out.write('\n')
                out.write('# for rotate %.4f degree \n'% degree)
                out.write('timestep %f\n'% timestep)
                out.write('run %d\n'% time_rotate)
                out.write('\n')
                out.write('unfix 32\n')
                out.write('unfix 33\n')
                out.write('reset_timestep 0\n')
            
            
            
            
            ###### fixed
            out.write('# fixed bp \n')
            # variable
            out.write('variable	Ax atom %s \n'% Ax)
            out.write('variable	Ay atom %s \n'% Ay)
            out.write('variable	Az atom %s \n'% Az)
            out.write('variable	Vx atom %s \n'% Vx)
            out.write('variable	Vy atom %s \n'% Vy)
            out.write('variable	Vz atom %s \n'% Vz)
            out.write('\n')            
            out.write('fix 12 fixdna move variable v_Ax v_Ay v_Az v_Vx v_Vy v_Vz \n')
            out.write('\n')

            out.write('comm_style tiled\n')
    #        out.write('comm_modify cutoff 2.5\n')
            out.write('fix 3 all balance 1000 1.05 rcb\n')
            out.write('\n')
            out.write('timestep %f\n'% timestep)
            out.write('\n')
            out.write('# output\n')
            out.write('compute quat all property/atom quatw quati quatj quatk\n')
            out.write('dump quat all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , outputfile_name))
            out.write('dump_modify quat sort id\n')
            out.write('compute erot dna erotate/asphere\n')
            out.write('compute ekin dna ke\n')
            out.write('compute peratom dna pe/atom\n')
            out.write('compute epot dna reduce sum c_peratom\n')
            out.write('variable erot equal c_erot\n')
            out.write('variable ekin equal c_ekin\n')
            out.write('variable epot equal c_epot\n')
            out.write('variable etot equal c_erot+c_ekin+c_epot\n')
            out.write('fix 5 all print ${efreq} "$(step)  ekin = ${ekin} |  erot = ${erot} | epot = ${epot} | etot = ${etot}" file %s\n' % outputenergy_name )
            out.write('\n')
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('\n')
            out.write('\n')
            out.write('\n')

            out.close()
        print(f'lammps infile done! total {num_file} files\n')

    #for shift
    @classmethod
    def in_file_generate_DNAbreak_21(self,in_file_name,read_file_name,dimension = 3, num_file = 1,T = 300,seq = 'seqdep',salt = 0.2,damp = 1.0,timestep = 0.01,time = 100000000,path=os.getcwd(),r = 2.0,v = 1/20000,bp_num = 336,fixed_bp_len = 1,degree= 0,first_end_d=0.5):
        temperature = T/3000
        output_freq = 100000
        energy_freq = 100000
        in_name = str(in_file_name) + '_break_T_' + str(T)+'_l_'+str(r)+'_r_'+str(degree)
    # ends parameter
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
        Vv = v #speed  for timestep = 0.002 output_freq=100000 1 frame = 0.01
        Ax = 'time*v_Vv'
        Ay = '0.0'
        Az = '0.0'
        Bx = '-time*v_Vv'
        By = '0.0'
        Bz = '0.0'
        distance = (r-first_end_d)/2 #left and right move the same time 
        time_push = int(distance/timestep/Vv)
    # endsfix_number
        bp = bp_num #double bps
        fix_bpnumber = fixed_bp_len
        dnanve = str(fix_bpnumber+1)+':'+str(bp-fix_bpnumber)+' '+ str(bp+1+fix_bpnumber) + ':' + str(2*bp-fix_bpnumber)
        moveleft = ''
        moveright = ''
        for i in range(fix_bpnumber):
            moveleft += str(bp-i) + ' ' + str(bp+i+1) + ' '
            moveright += str(i+1) + ' ' + str(2*bp-i) + ' '
        ll = '# lammps in file with supercoling with type 1 2 3 4 nve/dot\n'
        ll+= f'# total {bp_num} base pair\n'
        ll+= f'# fixed ends mode fixed {fixed_bp_len} bp of ends\n'
        ll+= f'# ends distance is begin {first_end_d} end {r}\n'
        ll+= f'# fixed ends mode rotation {degree} o of ends\n'
        for i in range(num_file):
            random_seed = random.randint(1, 50000) #set random for langevin
            outputfile_name = 'original_T_' +  str(T) + '_' + str(i) + '.lammpstrj' #set outputfile_name
            pre_outputfile_name = 'pre_' + outputfile_name
            outputenergy_name = outputfile_name + 'energy.txt'
            temperature_l = temperature #temperature for langevin
            out = open(os.path.join(path,str(in_name)+ '_' + str(i)),'w')
            out.write(ll)
            out.write('variable ofreq	equal %d\n' % output_freq)
            out.write('variable efreq	equal %d\n' % energy_freq)
            out.write('echo	screen\n')
            out.write('units lj\n')
            out.write('\n')
            out.write('dimension %d\n'% dimension)
            out.write('\n')
            out.write('newton on\n')
            out.write('\n')
            out.write('boundary  p p p\n')
            out.write('atom_style hybrid bond ellipsoid oxdna\n')
            out.write('atom_modify sort 0 1.0\n')
            out.write('\n')
            out.write('# Pair interactions require lists of neighbours to be calculated\n')
            out.write('neighbor 1.0 nsq\n')
            out.write('neigh_modify every 1 delay 0 check yes\n')
            out.write('\n')
            out.write('read_data %s\n'% read_file_name)
            out.write('set atom * mass 3.1575\n')
            out.write('\n')
            out.write('group all type 1 2 3 4\n')
            out.write('group dna type 1 2 3 4\n')
            out.write('group dnanve id %s\n'% dnanve)
            out.write('group moveleft id %s\n'% moveleft)
            out.write('group moveright id %s\n'% moveright)
            out.write('\n')
            out.write('# oxDNA bond interactions - FENE backbone\n')
            out.write('bond_style oxdna2/fene\n')
            out.write('bond_coeff * 2.0 0.25 0.7564\n')
            out.write('special_bonds lj 0 1 1\n')
            out.write('\n')
            out.write('# oxDNA pair interactions\n')
            out.write('pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh\n')
            out.write('pair_coeff * * oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32\n')
            out.write('pair_coeff * * oxdna2/stk     %s %.4f 1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65\n'% (seq,temperature))
            out.write('pair_coeff * * oxdna2/hbond   %s 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 1 4 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff 2 3 oxdna2/hbond   %s 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45\n'% seq)
            out.write('pair_coeff * * oxdna2/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68\n')
            out.write('pair_coeff * * oxdna2/coaxstk 58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793\n')
            out.write('pair_coeff * * oxdna2/dh      %.4f %.4f 0.815\n'% (temperature,salt))
            out.write('\n')
            out.write('# NVE ensemble\n')
            out.write('fix 1 dnanve nve/dotc/langevin %.4f %.4f %.4f %d angmom 10\n'% (temperature_l,temperature_l,damp,random_seed))
            out.write('\n')
            out.write('comm_style tiled\n')
    #        out.write('comm_modify cutoff 2.5\n') #to fix cutoff 
            out.write('fix 3 all balance 1000 1.05 rcb\n')
            out.write('# fixmove \n')
            out.write('variable	Vv equal %.8f \n'% Vv)
            out.write('variable	Ax atom %s \n'% Ax)
            out.write('variable	Ay atom %s \n'% Ay)
            out.write('variable	Az atom %s \n'% Az)
            out.write('variable	Bx atom %s \n'% Bx)
            out.write('variable	By atom %s \n'% By)
            out.write('variable	Bz atom %s \n'% Bz)
            out.write('variable	Vx atom %.4f \n'% Vx)
            out.write('variable	Vy atom %.4f \n'% Vy)
            out.write('variable	Vz atom %.4f \n'% Vz)
            out.write('fix 12 moveright move variable v_Ax v_Ay v_Az v_Vx v_Vy v_Vz \n')
            out.write('fix 13 moveleft move variable v_Bx v_By v_Bz v_Vx v_Vy v_Vz \n')

            out.write('\n')
            out.write('timestep %f\n'% timestep)
            out.write('\n')
    # pre
            out.write('# pre output\n')
            out.write('compute quat all property/atom quatw quati quatj quatk\n')
            out.write('dump quatt all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , pre_outputfile_name))
            out.write('dump_modify quatt sort id\n')
            out.write('\n')
            out.write('# for move %.4f distance \n'% r)
            out.write('run %d\n'% time_push)
    #unfix damp part
            out.write('\n')
            out.write('undump quatt\n')
            out.write('unfix 12\n')
            out.write('unfix 13\n')
            out.write('\n')
    #fix ends
            out.write('variable	Cx atom 0 \n')
            out.write('variable	Cy atom 0 \n')
            out.write('variable	Cz atom 0 \n')
            out.write('fix 22 moveright move variable v_Cx v_Cy v_Cz v_Vx v_Vy v_Vz \n')
            out.write('fix 23 moveright move variable v_Cx v_Cy v_Cz v_Vx v_Vy v_Vz \n')
            out.write('\n')
            out.write('# output\n')
            out.write('dump quat all custom %d %s id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n' % (output_freq , outputfile_name))
            out.write('dump_modify quat sort id\n')
            out.write('compute erot dna erotate/asphere\n')
            out.write('compute ekin dna ke\n')
            out.write('compute peratom dna pe/atom\n')
            out.write('compute epot dna reduce sum c_peratom\n')
            out.write('variable erot equal c_erot\n')
            out.write('variable ekin equal c_ekin\n')
            out.write('variable epot equal c_epot\n')
            out.write('variable etot equal c_erot+c_ekin+c_epot\n')
            out.write('fix 5 all print ${efreq} "$(step)  ekin = ${ekin} |  erot = ${erot} | epot = ${epot} | etot = ${etot}" file %s\n' % outputenergy_name )
            out.write('\n')
            out.write('reset_timestep 0\n')
            out.write('\n')
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
            out.write('run %d\n'% time)
    #        out.write('run %d\n'% time)
            out.write('\n')
            out.write('\n')
            out.write('\n')

            out.close()

        print(f'lammps infile done! total {num_file} files\n')


class Dna_array_system():
    num = 0
    sigma = 0.8518 # nm
    N_A = 6.023
    def __init__(self,box_size,seq=None, probe_sq_n = 6, target_n = 36,infile_n = 1,runflag = True, zmax = 20, fold_name = '1') -> None:
        Dna_array_system.num += 1
        self._box = np.array([-box_size,box_size,-box_size,box_size,-1,zmax+1],dtype=np.float64)
        self._seq = base.to_seq(sequence=seq)
        self._bp_len = len(self._seq)
        self._probe_sq_n = probe_sq_n
        self._probe_n = self._probe_sq_n *self._probe_sq_n 
        self._target_n = target_n
        # suface_density unit nm^-2
        self.surface_density = self._probe_sq_n**2/(4*box_size*box_size*Dna_array_system.sigma*Dna_array_system.sigma)
        # target_concentration unit mu M
        v_box = zmax*4*box_size*box_size*Dna_array_system.sigma*Dna_array_system.sigma*Dna_array_system.sigma # nm^3
        N_A = 6.023
        target_concentration = self._target_n/v_box/N_A*10000 #mol/m^3
        self.target_concentration = target_concentration * 1000 # mu M
        #the zpos of target array
        self._z_tem_wall = int(base.BASE_BASE*self._bp_len+0.5)+1
        system = StrandGenerator.gen_probe_target(self._box,self._probe_sq_n,self._z_tem_wall+1,n_target=self._target_n,seq=seq)
        cwd = os.getcwd()
        fold_path = os.path.join(cwd,str(fold_name))
        try:
            os.makedirs(fold_path)
        except:
            print(f'path :{fold_path} already have ')

        system.get_lammps_data_file("probe8type.data", "top.data",path = fold_path)
        path = os.path.join(fold_path,'probe8type_ovt.data')
        with open(path,'w+') as f:
            f.write(system.lammps_output)
        print('-----lammps generated done!')

        system.get_system_graph()
        del system

        Lammps_in_file.in_file_generate_with_relax_8type_fixwall(in_file_name = 'in.lk'  ,num_file = infile_n,read_file_name= 'probe8type.data',T = 300,salt = 0.5,n_probe=self._probe_n,n_target=self._target_n,bplen=self._bp_len,surface_density=self.surface_density,target_density=self.target_concentration,runflag=runflag,path=fold_path)
        
        print(f'path :{fold_path} done !')
        print(f'It is the {Dna_array_system.num}th fold!')
        print('-----------------------------------------')

class Dna_supercoiling_system():
    num = 0
    sigma = 0.8518 # nm
    N_A = 6.023
    def __init__(self,box_size=0,seq=None,delta_lk=0,infile_n = 1,nickflag = False,dump = 1.0, fold_name = 'Dna_supercoiling_system') -> None:
        Dna_supercoiling_system.num += 1
        self._box = np.array([-box_size,box_size,-box_size,box_size,-box_size,box_size],dtype=np.float64)
        self._seq = base.to_seq(sequence=seq,bp=len(seq))
        self._bp_len = len(self._seq)
        system = base.System(self._box)
        system.add_strands(StrandGenerator.generate(len(seq), sequence=seq, double=True, circular=True, DELTA_LK=delta_lk), check_overlap=False)
        if nickflag:
            system._strands[1].make_nick()
        system.update_probe_target_system(probe_mol= 1)
        top_message = system.get_tw_wr_lk(system._strands[0],system._strands[1])
        print('topmessage is ',top_message)
        cwd = os.getcwd()
        fold_path = os.path.join(cwd,str(fold_name))
        try:
            os.makedirs(fold_path)
        except:
            print(f'path :{fold_path} already have ')

        system.get_lammps_data_file("single.data", "top.data",path = fold_path)
        path = os.path.join(fold_path,'single_ovt.data')
        with open(path,'w+') as f:
            f.write(system.lammps_output)
        print('-----lammps generated done!')
        
        del system

        Lammps_in_file.in_file_generate_21(in_file_name = 'in.lk' ,read_file_name= 'single.data',num_file = infile_n,T = 300,damp=dump,path=fold_path)#,timestep=0.01,time=10000000,outputfre=10000)
        
        print(f'path :{fold_path} done !')
        print(f'It is the {Dna_supercoiling_system.num} th fold!')
        print('-----------------------------------------')

class Dna_break_endsfixed_pre():
    num = 0
    def __init__(self,box_size=0,seq=None,delta_lk=0,infile_n = 1,breakflag = True,dump = 78.9375, fold_name = 'Dna_break_endsfixed_pre',fixed_len=1,v1_l=1,rotate=0) -> None:
        Dna_break_endsfixed_pre.num += 1
        self._box = np.array([-box_size,box_size,-box_size,box_size,-box_size,box_size],dtype=np.float64)
        self._seq = base.to_seq(sequence=seq,bp=len(seq))
        self._bp_len = len(self._seq)
        system = base.System(self._box)
        #system.add_strands(StrandGenerator.generate_circle_fixedends(len(seq), sequence=seq, double=True, circular=False, DELTA_LK=delta_lk), check_overlap=False)
        system.add_strands(StrandGenerator.generate_circle_fixedends(len(seq), sequence=self._seq, double=True, circular=True, DELTA_LK=delta_lk,v1_l=v1_l,degree=rotate), check_overlap=False)

        if breakflag:
            system._strands[0].make_nick()
            system._strands[1].make_nick()
        system.update_probe_target_system(probe_mol= 1)

        top_message = system.get_tw_wr_lk(system._strands[0],system._strands[1])
        print('topmessage is ',top_message)
        cwd = os.getcwd()
        fold_path = os.path.join(cwd,str(fold_name))
        try:
            os.makedirs(fold_path)
        except:
            print(f'path :{fold_path} already have ')
        
        system.get_lammps_data_file("doublerelax.data", "toprelax.data",path = fold_path,message = f'break_relax_delta_lk={delta_lk} l={v1_l}')
        path = os.path.join(fold_path,'double_ovt.data')
        with open(path,'w+') as f:
            f.write(system.lammps_output)
        path = os.path.join(fold_path,'double_ovt_all.data')
        with open(path,'w+') as f:
            f.write(system.ovt_lammps_output)
        print('-----lammps generated done!')
        
        del system

        Lammps_in_file.in_file_generate_21_break_fixed_ends_plus_rotation(in_file_name = 'realax.lk' ,read_file_name= 'doublerelax.data',num_file = infile_n,T = 300,damp=dump,path=fold_path,bp_num = self._bp_len,fixed_bp_len = fixed_len,degree=rotate,first_end_d=v1_l)#,time=10000000,outputfre=10000)
    


        print(f'path :{fold_path} done !')
        print(f'It is the {Dna_break_endsfixed_pre.num} th fold!')
        print('-----------------------------------------')

class DNA_break_endsfixed():
    num = 0
    def __init__(self,box_size=0,seq=None,delta_lk=0,infile_n = 1,breakflag = True,dump = 2.5, fold_name = 'Dna_break_endsfixed',fixed_len=1,v1_l=1,rotate=0,r=-1) -> None:
        DNA_break_endsfixed.num += 1
        self._box = np.array([-box_size,box_size,-box_size,box_size,-box_size,box_size],dtype=np.float64)
        self._seq = base.to_seq(sequence=seq,bp=len(seq))
        self._bp_len = len(self._seq)
        cwd = os.getcwd()
        fold_path = os.path.join(cwd,str(fold_name))
        try:
            os.makedirs(fold_path)
        except:
            print(f'path :{fold_path} already have ')
        Lammps_in_file.in_file_generate_DNAbreak_21(in_file_name = 'in.lk' ,read_file_name= 'double.data',num_file = infile_n,T = 300,damp=dump,path=fold_path,bp_num = self._bp_len,fixed_bp_len = fixed_len,degree=rotate,first_end_d=v1_l,r=r)#,time=10000000,outputfre=10000)
        print(f'path :{fold_path} done !')
        print(f'It is the {Dna_break_endsfixed_pre.num} th fold!')
        print(f'start len is {v1_l} end len is {r}')
        print('-----------------------------------------')

def dnaarray():
    srepeat=['AAAAAAAAAAAAAAAAAAAA',
    'ACACACACACACACACACAC',
    'CCCCCCCCCCCCCCCCCCCC']
    boxsize = [9,15,30,60]#,12,9]
    # for i in range(3):
    #     for j in boxsize:
    #         Dna_array_system(box_size = j,seq=srepeat[i][::-1],infile_n=20,runflag=True,fold_name=j*100+i)

    for i in [2]:
        for j in[30]:
            for k in [3,10,20]:
                Dna_array_system(box_size = j,probe_sq_n=k,seq=srepeat[i][::-1],infile_n=20,runflag=True,fold_name=(j*100+i)*100+k)    
    # double = StrandGenerator()
    # s = base.System([-200,200,-200,200,-200,200])
    # pwd = os.getcwd() 
    # filename = 'single.data'
    
    # # open file
    # w = base.File_system(pwd,filename)
    # seq = 'C' * 336
    # success = s.add_strands(double.generate(len(seq), sequence=seq, double=True, circular=False, DELTA_LK=4), check_overlap=False)
    # s.update_probe_target_system(probe_mol= 1)

    # s.get_lammps_data_file()

    # s.get_hb_pair(mode = 'all')

    # with open('probe8type_ovt.data','w+') as f:
    #     f.write(s.lammps_output)
    # print('-----lammps generated done!')
    # del w
    # del s
    pass

def supercoiling():
    # i = int((336-42)/2)
    # seq = 'AT' * int(i)
    # Dna_supercoiling_system(box_size=200,seq=seq,delta_lk=3,infile_n=2,nickflag= False,dump=78.9375,fold_name='circledump'+str(len(seq))+'bpA')

    # i = int((336+42)/2)
    # seq = 'AT' * int(i)
    # Dna_supercoiling_system(box_size=200,seq=seq,delta_lk=3,infile_n=2,nickflag= False,dump=78.9375,fold_name='circledump'+str(len(seq))+'bpA')

    # exit('0')
    # seq = 'C' * 336
    # for i in range(29,36,1):
    #     Dna_supercoiling_system(box_size=200,seq=seq,delta_lk=i-32,infile_n=20,nickflag= True,fold_name='circlefash'+str(i))
    # pass
    # i = int(336/2)
    # seq = 'C' * 336
    # for i in [-1,-2,-3,-4,-5]:
    #     Dna_supercoiling_system(box_size=200,seq=seq,delta_lk=i,infile_n=2,nickflag= False,dump=78.9375,fold_name='circledump'+str(i+32))
    # Dna_supercoiling_system(box_size=200,seq='C' * 336,delta_lk=0,infile_n=1,nickflag= True,fold_name='circle'+str(111))
    cwd = os.getcwd()
    fold_path = os.path.join(cwd,'wr29')
    try:
        os.makedirs(fold_path)
    except:
        print(f'path :{fold_path} already have ')
    Lammps_in_file.in_file_generate_21(in_file_name = 'in.lk' ,read_file_name= 'single.data',num_file = 30,T = 300,damp=1.0,path=fold_path)

def dnabreak():
    # Lammps_in_file.in_file_generate_DNAbreak_21(in_file_name = 'in.lk'  ,num_file = 5,read_file_name= 'double.data',T = 300,r=4,timestep = 0.01)
    seq = 'A' * 336
    rot = 45
    len_l = 2
    for i in range(35,36,1):
        Dna_break_endsfixed_pre(box_size=200,seq=seq,delta_lk=i-32,infile_n=4,breakflag=True,fold_name=f'break_pre_lk_{i}_l_{len_l}r_{rot}',fixed_len=1,v1_l=len_l,rotate=rot)
        DNA_break_endsfixed(box_size=200,seq=seq,r=5,dump=2.5,v1_l=len_l,rotate=rot,infile_n=32,fold_name=f'break_lk_{i}_l_{len_l}r_{rot}',fixed_len=1)
        break
if __name__ == "__main__":
#    dnaarray()
    # supercoiling()
    dnabreak()
#######################supercoiling in file just relax ###########dump = 1.0 for supercoling ,2.5 for hybrid ,78.9375 for sampling
    # Lammps_in_file.in_file_generate_21(in_file_name = 'in.lk' ,read_file_name= 'single.data',num_file = 20,T = 300)


    pass
