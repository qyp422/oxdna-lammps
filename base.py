'''
Author: qyp422
Date: 2022-10-13 16:21:26
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-09-18 20:31:41
Description: system meassage

Copyright (c) 2022 by qyp422, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import sys,os
import scipy
import pair_oxdna2_hbond as hb

import mathfuction as mf

import matplotlib as mpl
import matplotlib.pyplot as plt
fontbase = {'family':'Arial','weight':'normal','size':18}
plt.rc('font',**fontbase)

import networkx as nx

print('\n' + __file__ + ' is called\n')

number_to_base = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'T'}


base_to_number = {'A' : 0, 'a' : 0, 'C' : 1, 'c' : 1,
                  'G' : 2, 'g' : 2, 'T' : 3, 't' : 3,
                  'U' : 3, 'u' : 3, 'D' : 4}

def to_seq(sequence=None,bp=20):
    if isinstance(sequence, list):
        sequence = np.array(sequence)
        return sequence
    # Loads of input checking...
    if isinstance(sequence, str):
        try:
            sequence = np.array([base_to_number[x] for x in sequence])
            return sequence
        except:
            sys.exit("Key Error: sequence is invalid")
            
    if sequence == None:
        sequence = np.random.randint(0, 4, bp)
    elif len(sequence) != bp:
        n = bp - len(sequence)
        sequence = np.append(sequence, np.random.randint(0, 4, n))
        print("sequence is too short, adding %d random bases" % n)
    return sequence

try:
    FLT_EPSILON = np.finfo(np.float64).eps
except:
    FLT_EPSILON = 2.2204460492503131e-16

H_CUTOFF = -0.1 #hb energy cut off 

POS_BACK = -0.4
POS_MM_BACK1 = -0.3400
POS_MM_BACK2 = 0.3408
POS_STACK = 0.34
POS_BASE = 0.4
CM_CENTER_DS = POS_BASE + 0.2
BASE_BASE = 0.3897628551303122


FENE_EPS = 2.0
FENE_R0_OXDNA2 = 0.7564




class Nucleotide():
    """
    Nucleotides compose Strands

    cm_pos --- Center of mass position
        Ex: [0, 0, 0]

    a1 --- Unit vector indicating orientation of backbone with respect to base
        Ex: [1, 0, 0]

    a3 --- Unit vector indicating orientation (tilting )of base with respect to backbone
        Ex: [0, 0, 1]

    base --- Identity of base, which must be designated with either numbers or

        oxdna
            letters (this is called type in the c++ code). Confusingly enough, this
            is similar to Particle.btype in oxDNA.
            
            Number: {0,1,2,3} or any number in between (-inf,-7) and (10, inf)
            To use specific sequences (or an alphabet large than four) one should
            start from the complementary pair 10 and -7. Complementary pairs are
            such that base_1 + base_2 = 3;
            
            Letter: {A,G,T,C} (these will be translated to {0, 1, 3, 2}).

        lammps s
            equence-specific base-pairing strength
            A:0 4  C:1 5  G:2 6  T:3 7, 5'- [i][j] -3'

        These are set in the dictionaries: number_to_base, base_to_number
    
    btype--- Identity of base. Unused at the moment.

    """
    index = 0

    def __init__(self,cm_pos,quat,base,n3=-1,n5=-1,next=-1):
        self.index = Nucleotide.index
        Nucleotide.index += 1
        self.cm_pos = np.array(cm_pos,dtype=float)
        self._quat = np.array(quat,dtype=float)
        a1,a2,a3 = mf.q_to_exyz(self._quat)

        self._a1 = np.array (a1)
        self._a2 = np.array (a2)
        self._a3 = np.array (a3)
        self._base = base
        self.strand = -1        #
        self.n3 = n3            #   front nt_id
        self.n5 = n5            #   next nt_id
        self.next = next        # for cell list
        self.interactions = []  #what other nucleotide this nucleotide actually interacts with
        # self.init_interactions()
    def __str__(self):
        return f'{self.index} {self.strand} {self._base} {self.cm_pos[0]:.6} {self.cm_pos[1]:.6} {self.cm_pos[2]:.6} {self._quat[0]:.6} {self._quat[1]:.6} {self._quat[2]:.6} {self._quat[3]:.6}\n'
    def copy(self, disp=None, rot=None):
        copy = Nucleotide(self.cm_pos, self._quat, self._base)
        # if disp is not None:
        #     copy.translate(disp)
        # if rot is not None:
        #     copy.rotate(rot)
        return copy

    def get_pos_base (self):
        """
        Returns the position of the base centroid
        Note that cm_pos is the centrod of the backbone and base.
        """
        return self.cm_pos + self._a1 * POS_BASE

    pos_base = property(get_pos_base)

    def get_center_ds(self):
        return self.cm_pos + self._a1 * CM_CENTER_DS

    pos_center_ds = property(get_center_ds)

    def get_pos_stack (self):
        return self.cm_pos + self._a1 * POS_STACK

    pos_stack = property (get_pos_stack)

    def get_pos_back (self):
        """
        Returns the position of the backbone centroid
        Note that cm_pos is the centrod of the backbone and base.
        """
        return self.cm_pos + self._a1 * POS_MM_BACK1 + self._a2 * POS_MM_BACK2

    pos_back = property (get_pos_back)


    def get_draw_networkx(self):
        # pos backbone xy
        tem = self.cm_pos + self._a1 * POS_MM_BACK1 + self._a2 * POS_MM_BACK2
        return tem[0],tem[1]
    
    pos_back_xy = property (get_draw_networkx)

    def distance (self, other, PBC=True, box=None):
        if PBC and box is None:
            if not (isinstance (box, np.ndarray) and len(box) == 3):
                print ("distance between nucleotides: if PBC is True, box must be a numpy array of length 3")
        dr = other.pos_back - self.pos_back
        if PBC:
            dr -= box * np.rint (dr / box)
        return dr



    def get_lammps_output(self):
        return f'{self.index+1} {self.strand+1} {self._base+1} {self.cm_pos[0]:.6f} {self.cm_pos[1]:.6f} {self.cm_pos[2]:.6f} {self._quat[0]:.6f} {self._quat[1]:.6f} {self._quat[2]:.6f} {self._quat[3]:.6f}\n'
            
    def get_ovt_lammps_output(self):
        # position of sugar-phosphate backbone interaction site in oxDNA2
        cm_back = self.cm_pos + POS_MM_BACK1 * self._a1 + POS_MM_BACK2 * self._a2
        # position of base interaction site in oxDNA2
        cm_base = self.cm_pos + POS_BASE * self._a1

        ovt = f'{self.index+1} {self.strand+1} {self._base+1} {cm_back[0]:.6f} {cm_back[1]:.6f} {cm_back[2]:.6f} {self._quat[0]:.6f} {self._quat[1]:.6f} {self._quat[2]:.6f} {self._quat[3]:.6f}'
        ovt += ' 0.4 0.4 0.4\n'
        ovt += f'{-self.index-1} {-self.strand-1} {self._base+11} {cm_base[0]:.6f} {cm_base[1]:.6f} {cm_base[2]:.6f} {self._quat[0]:.6f} {self._quat[1]:.6f} {self._quat[2]:.6f} {self._quat[3]:.6f}'
        ovt += ' 0.5 0.2 0.1\n'
        return ovt

class Strand():
    """
    Strand composed of Nucleotides
    Strands can be contained in System
    """
    index = 0

    def __init__(self):
        self.index = Strand.index
        Strand.index += 1
        self._first = -1            #first nt.index                                        
        self._last = -1             #last nt.index
        self._n = 0                 #strand length
        self._nucleotides = []      #all nt


        self._sequence = []         #seq
        self.visible = False
        self.H_interactions = {}    #shows what strands it has H-bonds with
        self.H_interaction_number = 0
        self._circular = False      #bool on circular DNA

    def add_nucleotide(self, n:Nucleotide):
        if len(self._nucleotides) == 0:
            self._first = n.index
        
        n.strand = self.index
        self._n += 1
        self._nucleotides.append(n)
        self._last = n.index
        self._sequence.append(n._base)
    
    def __str__(self):
        return ''.join([str(x) for x in self._nucleotides])

    def copy(self):
        copy = Strand()
        for n in self._nucleotides:
            copy.add_nucleotide(n.copy())
        return copy

    def append (self, other):
        if not isinstance (other, Strand):
            raise ValueError

        dr = self._nucleotides[-1].distance (other._nucleotides[0], PBC=False)
        if np.sqrt(np.dot (dr, dr)) > (0.7525 + 0.25):
            print >> sys.stderr, "WARNING: Strand.append(): strands seem too far apart. Assuming you know what you are doing."

        ret = Strand()

        for n in self._nucleotides:
            ret.add_nucleotide(n)

        for n in other._nucleotides:
            ret.add_nucleotide(n)

        return ret

    def get_slice(self, start=0, end=None):
        if end is None: end = len(self._nucleotides)
        ret = Strand()
        for i in range(start, end):
            ret.add_nucleotide(self._nucleotides[i].copy())
        return ret

    def change_strand_index(self,index:int):
        for n in self._nucleotides:
            n.strand = index
        self.index = index

    def get_center_of_mass(self):
        cm = np.zeros(3)
        for n in self._nucleotides:
            cm += n.cm_pos
        return cm / self._n
    
    def get_sor(self):
        sor = 0.0
        for n in self._nucleotides:
            sor += 1.5 * n._a3[2] * n._a3[2] - 0.5
        return sor/self._n
    
    def get_rgrsq(self):
        rgrsq = np.zeros(3)
        cm_strand = self.cm
        for n in self._nucleotides:
            rgrsq += (n.cm_pos-cm_strand)*(n.cm_pos-cm_strand)
        return rgrsq/self._n


    cm = property (get_center_of_mass)
    sor = property (get_sor)
    rgrsq = property (get_rgrsq)

    def add_H_interaction(self,other_strand):
        self.H_interaction_number += 1
        if other_strand in self.H_interactions:
            self.H_interactions[other_strand] += 1
        else:
            self.H_interactions[other_strand] = 1
    
    def make_circular(self, check_join_len=False):
        if check_join_len:
            dr = self._nucleotides[-1].distance(self._nucleotides[0], PBC=False)
            if np.sqrt(np.dot (dr, dr)) > (0.7525 + 0.25):
                print("Strand.make_circular(): ends of the strand seem too far apart. \
                            Assuming you know what you are doing.")
        self._circular = True

    def make_nick(self,rotation_number=False):
        self._circular = False

    # output

    def get_lammps_output(self):
        return ''.join([n.get_lammps_output() for n in self._nucleotides])

    def get_ovt_lammps_output(self):
        return ''.join([n.get_ovt_lammps_output() for n in self._nucleotides])

    def get_lammps_N_of_bonds_strand(self):
        N_bonds = 0
        for n in self._nucleotides:
            if n.index != self._last:
                N_bonds += 1
            elif self._circular:
                N_bonds += 1

        return N_bonds

    def get_lammps_bonds(self):
        top = []
        for n in self._nucleotides:
            if n.index != self._last:
                top.append("%d  %d" % (n.index + 1, n.index + 2))
            elif self._circular:
                top.append("%d  %d" % (n.index + 1, self._first + 1))

        return top


# after system build please run  self.update_nucleotide_list() and self.map_nucleotides_to_strands()
class System():
    read_count = 0
    T = 0.1
    salt_concentration = 1.0
    qeff_dh_one = 0.815
    def __init__(self,box,time=0):
        self.frame_path = os.path.join(os.getcwd(),'frame_path')
        try:
            os.makedirs(self.frame_path)
        except:
            pass
        #system parameter
        self._time = time
        self._box = np.array(box,np.float64)
        self.l_box = np.array([box[1] - box[0],box[3] - box[2],box[5] - box[4]],np.float64)
        self.inedx = self.read_count
        self.read_count = System.read_count
        System.read_count += 1
        # 

        self._N = 0             #system nt number
        self._N_strands = 0     #system nstrands number
        self._strands = []      #list to save strands
        self._nucleotides = []  #list to save nucleotides
        self._nucleotide_to_strand = [] #nucleotide id to strand id 
        cells_done = False

        # hb analysis /array sysytem
        # self._hb_number = 0     #system h bond pair number
        # self._hb_probe_number = 0   #system probe number to be hb
        # self._hb_target_number = 0  #system target number to be hb
        self._hb_dict = {}          #system hb pair dict only _hb_dict[probe id] = target id
        self._hb_energy = False     #system_hb_energy

        self.probe_mol = False      #strands number before probe_n is probe after is target
        self.target_mol = False     #*_mol is the strands number of *
        self.probe_n = False        #*_n is the nt number of *
        self.target_n =False        #* is the index of * id
        self.probe = False
        self.target = False         #* = probe / target

        
    def get_sequences (self):
        return [x._sequence for x in self._strands]
    _sequences = property (get_sequences)
    def __str__(self):
        return ''.join([str(x) for x in self._strands])
    


    '''
    description: 
    add single strand
    '''
    def add_strand(self, s : Strand, check_overlap=False):
        """

        Add a Strand to the System

        Returns True if non-overlapping
        Returns False if there is overlap
        will be open this function
        """
        pass 
        '''
        # we now make cells off-line to save time when loading
        # configurations; interactions are computed with h_bonds.py
        # most of the time anyways
        for n in s._nucleotides:
            cs = np.array((np.floor((n.cm_pos/self._box - np.rint(n.cm_pos / self._box ) + 0.5) * (1. - FLT_EPSILON) * self._box / self._cellsides)), np.int)
            cella = cs[0] + self._N_cells[0] * cs[1] + self._N_cells[0] * self._N_cells[1] * cs[2]
            n.next = self._head[cella]
            self._head[cella] = n
        '''
        self._strands.append(s)
        self._N += s._n
        self._N_strands += 1
        self.cells_done = False
        return True
    '''
    description: 
    add strands
    '''
    def add_strands(self, ss, check_overlap=False):
        if isinstance(ss, tuple) or isinstance(ss, list):
            added = []
            for s in ss:
                if self.add_strand(s, check_overlap):
                    added.append(s)
            if len(added) == len(ss):
                return True
            else:
                for s in added:
                    Nucleotide.index -= s.n
                    Strand.index -= 1
                    self._strands.pop()
                    self._N -= s.n
                    self._N_strands -= 1
                    self._sequences.pop()
                return False
        elif isinstance(ss, Strand):
            if self.add_strand(ss, check_overlap):
                return True
            else:
                Nucleotide.index -= s.n
                Strand.index -= 1
                self._strands.pop()
                self._N -= s.n
                self._N_strands -= 1
                self._sequences.pop()
                return False
        return False




    '''
    description: 
    refresh_system
    the same as del all strands
    '''
    def refresh_system(self):
        self._strands=[]
        self._N = 0
        self._N_strands = 0
        self.cells_done = False

    def map_nucleotides_to_strands(self):
        #this function creates nucl_id -> strand_id array
        for i in range(self._N_strands):
            for j in range(self._strands[i]._n):
                self._nucleotide_to_strand.append(i)
        print('updata ._nucleotide_to_strand done!!!')

    # get list of nucleotide
    def update_nucleotide_list (self):
        self._nucleotides = []
        for s in self._strands:
            self._nucleotides += s._nucleotides
        print('updata ._nucleotide done!!!')
        
    # update parameter note: use update_nucleotide_list fisrt 
    def update_nucleotide_pos_quat(self,nucleotide_id,cm=False,base=False,quat=False,n3=False,n5=False):
        if len(self._nucleotides) != self._N:
            sys.exit(r'please use self.update_nucleotide_list updata self._nucleotides first\n' + f'{len(self._nucleotides)} {self._N}')
        nt = self._nucleotides[nucleotide_id]
        if cm:
            nt.cm_pos = np.array(cm,dtype=float)
        if quat:
            nt._quat = np.array(quat,dtype=float)
            a1,a2,a3 = mf.q_to_exyz(nt._quat)
            nt._a1 = np.array (a1)
            nt._a2 = np.array (a2)
            nt._a3 = np.array (a3)
        if base:
            nt._base = base
        if n3:
            nt.n3 = n3       
        if n5:
            nt.next = next

    def change_seq(self,seq):
        
        if len(self._nucleotides) != self._N:
            sys.exit(r'please use self.update_nucleotide_list updata self._nucleotides first\n' + f'{len(self._nucleotides)} {self._N}')
        if len(seq) != self._N:
            sys.exit(r'please give enough len seq\n' + f'{len(seq)} {self._N}')
        seq = to_seq(sequence=seq,bp=len(seq))
        for i in range(self._N):
            self.update_nucleotide_pos_quat(i,base=seq[i])
        print('seq change done!___________')

    def system_contactmap(self):
        pass


    def get_center_of_mass(self):
        return [s.cm for s in self._strands]
    
    def get_sor(self):
        return [s.sor for s in self._strands]
    
    def get_rgrsq(self):
        rg_list = [s.rgrsq for s in self._strands]
        return rg_list

    def get_hb_pair(self,mode = 'all'):
        '''
        mode : all -- search every hb
               search -- search base on last hb dict 
        '''
        if len(self._nucleotide_to_strand) != self._N :
            self.map_nucleotides_to_strands()
        if len(self._nucleotides) != self._N :
            self.update_nucleotide_list()
        

        # refresh the hb parameter except hb_dict
        for s in self._strands:
            s.H_interactions = {}
            s.H_interaction_number = 0

        # refresh the hb total_energy    
        self._hb_energy = 0.0

        # start find hb bond
        hb_pair = {}
        if mode == 'search':
            probe_set = set(self.probe)
            target_set = set(self.target)
            # check if last frame hb
            for i in self._hb_dict:
                j = self._hb_dict[i]
                a = self._nucleotides[i]
                b = self._nucleotides[j]
                tem_hb_energy = hb.hb_energy(a.cm_pos,a._a1,a._a3,a._base,b.cm_pos,b._a1,b._a3,b._base,self.l_box) 
                self._hb_energy += tem_hb_energy
                if tem_hb_energy < H_CUTOFF:
                    hb_pair[i]=j
                    self._strands[self._nucleotide_to_strand[i]].add_H_interaction(self._nucleotide_to_strand[j])
                    self._strands[self._nucleotide_to_strand[j]].add_H_interaction(self._nucleotide_to_strand[i])
                    probe_set.discard(i)
                    target_set.discard(j)
                    

            # check remain probe
            for i in probe_set:
                a = self._nucleotides[i]
                for j in target_set:
                    b = self._nucleotides[j]
                    tem_hb_energy = hb.hb_energy(a.cm_pos,a._a1,a._a3,a._base,b.cm_pos,b._a1,b._a3,b._base,self.l_box) 
                    self._hb_energy += tem_hb_energy
                    if tem_hb_energy < H_CUTOFF:
                        hb_pair[i]=j
                        self._strands[self._nucleotide_to_strand[i]].add_H_interaction(self._nucleotide_to_strand[j])
                        self._strands[self._nucleotide_to_strand[j]].add_H_interaction(self._nucleotide_to_strand[i])
                        break
                if i in hb_pair:
                    target_set.discard(hb_pair[i])

        elif mode == 'all' :
            twin_hb_pair = {}
            probe_set = set(range(self._N))
            target_set = set(range(self._N))
            for i in probe_set:
                if (i in hb_pair) or (i in twin_hb_pair):
                    continue
                a = self._nucleotides[i]
                for j in target_set:
                    if i == j:
                        continue
                    b = self._nucleotides[j]
                    tem_hb_energy = hb.hb_energy(a.cm_pos,a._a1,a._a3,a._base,b.cm_pos,b._a1,b._a3,b._base,self.l_box) 
                    self._hb_energy += tem_hb_energy
                    if tem_hb_energy < H_CUTOFF:
                        hb_pair[i]=j
                        twin_hb_pair[j] = i
                        self._strands[self._nucleotide_to_strand[i]].add_H_interaction(self._nucleotide_to_strand[j])
                        self._strands[self._nucleotide_to_strand[j]].add_H_interaction(self._nucleotide_to_strand[i])
                        break
                if i in hb_pair:
                    target_set.discard(i)
                    target_set.discard(hb_pair[i])

        self._hb_dict = hb_pair
        print(f'total hb_energy = {self._hb_energy}')
        print('updata _hb_dict done')

    def get_total_hb_energy(self):
        return self._hb_energy

    total_hb_energy = property (get_total_hb_energy)

    def get_i_density_distribution(self,i = 2):
        # i = 0 x
        # i = 1 y
        # i = 2 z
        # x = [n.cm_pos[i]  for n in self._nucleotides if (n.index >= self.probe_n)]
        # y = [n.cm_pos[i]  for n in self._nucleotides if (n.index < self.probe_n)]
        return [n.cm_pos[i]  for n in self._nucleotides]
            
            

# about probe target system

    '''
    description: 
    param {*} self
    param {*} probe_mol after probe_mol is probe else is target
    param {*} probe_n for speed up
    return {*}
    '''
    def update_probe_target_system(self,probe_mol,probe_n=None):
        self.probe_mol = probe_mol
        self.target_mol = self._N_strands - probe_mol
        if not probe_n:
            probe_n = 0 
            for i in range(probe_mol):
                probe_n += self._strands[i]._n
        self.probe_n = probe_n
        self.target_n = self._N-probe_n
        self.probe = list(range(probe_n))
        self.target = list(range(probe_n,self._N))
        print('update probe target done!')

    def get_hb_percentage(self):
        hb_percentage_list = np.array([float(s.H_interaction_number)/s._n for s in self._strands])
        hb_mol_number_list = [len(s.H_interactions) for s in self._strands ]
        hb_number = float(len(self._hb_dict))
        tem = np.where(hb_percentage_list>0,1,0)
        probe_mol_n = int(np.sum(tem[:self.probe_mol]))
        target_mol_n = int(np.sum(tem[self.probe_mol:]))
        if probe_mol_n:
            probe_percentage  = sum(hb_percentage_list[:self.probe_mol]) / probe_mol_n
        else:
            probe_percentage = 0.0
        if target_mol_n:
            target_percentage = sum(hb_percentage_list[self.probe_mol:]) / target_mol_n
        else:
            target_percentage = 0.0
        return [hb_number/self.probe_n , float(probe_mol_n)/self.probe_mol , float(target_mol_n)/self.target_mol , probe_percentage , target_percentage],hb_percentage_list.tolist(),hb_mol_number_list

    def decision_structure(self):
        """
        structure_array: array
        0   free 
        1   Type 1A perfectmatch    
        2   Type 1B
        3   Type 2A
        4   Type 2B
        5   Type 3
        6   purely misaligned
        7   pseudoknot structure
        8   else
        """
        structure_array = [0,0,0,0,0,0,0,0,0]
        target_search_set = set(range(self.probe_mol,self._N_strands))
        
        while len(target_search_set):
            cluster,probes_count,total_degree,max_degree_of_target = self.cluster_dfs(target_search_set.pop())
            target_search_set = target_search_set - cluster
            # decide
            if probes_count == 0:          # free
                structure_array[0] += 1
            elif max_degree_of_target > 2: # Type 3 
                structure_array[5] += 1
            elif probes_count == 1:
                if len(cluster) == 2: # Type 1A or purely misaligned or pseudoknot structure
                    structure_type = self.one_to_one_hybrid_type(cluster)
                    structure_array[structure_type] += 1
                
                elif len(cluster) == 3: # Type 1B
                    structure_array[2] += 1
                else:
                    structure_array[8] += 1
                    print('1 probe error!')
            elif probes_count > 1:
                if total_degree == 2*len(cluster) -2 : # Type 2B
                    structure_array[4] += 1
                elif total_degree == 2*len(cluster): # Type 2A
                    structure_array[3] += 1
                else:
                    structure_array[8] += 1
                    print('2 or more probes error!')
            else :
                structure_array[8] += 1
                print('probes number error!')
        print(structure_array)
        print('decision_structure done!')
        return structure_array

    def cluster_dfs(self,x):
        visited = set()
        total_degree = 0
        max_degree_of_target = -1
        probes_count = 0
        wait_to_visit = set([x])
        while len(wait_to_visit) > 0:
            t_visit = wait_to_visit.pop()
            visited.add(t_visit)
            wait_to_visit.update(self._strands[t_visit].H_interactions.keys())
            wait_to_visit = wait_to_visit - visited
            
            #update degree
            t_visit_degree = len(self._strands[t_visit].H_interactions)

            if t_visit < self.probe_mol:
                probes_count += 1
            else:
                max_degree_of_target = max(max_degree_of_target,t_visit_degree)
            # print(wait_to_visit,visited)

            
            total_degree += t_visit_degree
        return visited,probes_count,total_degree,max_degree_of_target

    def one_to_one_hybrid_type(self,cluster): #1or6or7or8
        if isinstance(cluster,set):
            cluster = list(cluster)
        elif isinstance(cluster,int):
            for j in self._strands[cluster].H_interactions:
                cluster = [cluster,j]
                break
        cluster.sort()
        s_probe = self._strands[cluster[0]]
        s_target = self._strands[cluster[1]]
        register = {}
        last_register = s_probe._n * 2 + 2
        register_type = 0
        for i in range(s_probe._first,s_probe._last+1):
            if i in self._hb_dict:
                tem_register = s_target._last - self._hb_dict[i] - i + s_probe._first
                if tem_register in register:
                    register[tem_register] += 1
                else:
                    register[tem_register] = 1
                # print(i,self._hb_dict[i],tem_register)
                if abs(last_register - tem_register) > 2:
                    register_type += 1
                last_register = tem_register
            else:
                continue
#        print(register)
        if register_type == 2:
            # print(register,'7777777777777')
            return 7
        elif register_type == 1:
            register_total = 0
            register_number = 0
            for i in register:
                register_number += register[i]
                register_total += register[i]*i
            if abs(register_total) <= ((s_target._n+1) // 3 * register_number):
                return 1
            else:
                # print(register,'66666666666666666')
                return 6
        # print(register,'88888888888888')
        return 8
# about supercoiling system

    @classmethod
    def get_tw_wr_lk(self,strand1:Strand,strand2:Strand):

        tw = 0.0
        wr = 0.0
        lk = 0.0

        cir = strand2._circular
        s1 , s = mf.get_base_spline(strand1,circle_flag=cir)
        s2 , s = mf.get_base_spline(strand2,circle_flag=cir,reverse=True)
        tw = mf.get_sayar_twist(s1,s2,s[0],s[1],circular = cir)
        wr = mf.get_sayar_writhe(s1,s[0],s[1], splines2 = s2,circular = cir)
        # l_t = get_lenth_dsDNA(s1,0,s, splines2 = s2)
        lk = tw + wr

        return [tw,wr,lk]
    
    @classmethod
    def get_local_twist(self,strand1:Strand,strand2:Strand):
        cir = strand2._circular
        
        return mf.get_loacl_twist(strand1,strand2,circle=cir)

    def single(self):
        return [1 if x in self._hb_dict else 0 for x in range(self._strands[0]._n)]

    def get_circle_rotation(self,bp_first = 0):
        '''
        help circle system rotation bp_first bp
        let the new first bp index 0 is original bp_first index bp_first
        return new system
        note old index is wrong
        Nucleotide.index = 0
        Strand.index += 0
        '''
        if self._strands[0]._circular and self._strands[1]._circular:
            s= System(self._box, time=self._time)
            Nucleotide.index = 0
            # 
            l1 = len(self._strands[0]._nucleotides)
            l2 = len(self._strands[1]._nucleotides)
            if l1 != l2:
                print('warning may wrong answer!')

            s1 = self._strands[0].get_slice(start = bp_first)
            s2 = self._strands[0].get_slice(start = 0,end = bp_first)

            ss1 = s1.append(s2)
            ss1.change_strand_index(0)
            ss1.make_circular()
            s.add_strand(ss1)

            s3 = self._strands[1].get_slice(start = l2 - bp_first)
            s4 = self._strands[1].get_slice(start = 0,end = l2 - bp_first)

            ss2 = s3.append(s4)
            ss2.change_strand_index(1)
            s.add_strand(ss2)
            return s
        else:
            print('not all strands circle')
            return False

    @classmethod
    def get_contactmap(self,strand1:Strand,strand2:Strand):
        # no pcb
        n = len(strand1._nucleotides)
        l2 = len(strand2._nucleotides)
        
        if n != l2:
            print('not the same lenth of two strands!')
            return False

        center_line = []

        for n1 in range(n):
            center_line.append((strand1._nucleotides[n1].cm_pos+strand2._nucleotides[n-1-n1].cm_pos)/2)

        distance = np.empty((n, n),dtype=float)
        for i in range(n):
            for j in range(n):
                if i==j:
                    distance[i][j] = 0.0
                elif j > i:
                    distance[i][j] = np.linalg.norm(center_line[i]-center_line[j])
                elif i > j:
                    distance[i][j] = distance[j][i]
        # np.savetxt('a.txt',np.rint(distance),fmt='%d')
        return distance



    cm = property (get_center_of_mass)
    sor = property (get_sor)
    rgrsq = property(get_rgrsq)
    hb_percentage = property(get_hb_percentage)


    # output
    def get_lammps_output(self):
        conf = 'ITEM: TIMESTEP\n'
        conf += f'{self._time}\n'
        conf += 'ITEM: NUMBER OF ATOMS\n'
        conf += f'{self._N}\n'
        conf += 'ITEM: BOX BOUNDS pp pp pp\n'
        conf += f'{self._box[0]} {self._box[1]}\n'
        conf += f'{self._box[2]} {self._box[3]}\n'
        conf += f'{self._box[4]} {self._box[5]}\n'       
        conf += 'ITEM: ATOMS id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4]\n'
        
        return conf + ''.join([s.get_lammps_output() for s in self._strands])

    def get_ovt_lammps_output(self):
        conf = 'ITEM: TIMESTEP\n'
        conf += f'{self._time}\n'
        conf += 'ITEM: NUMBER OF ATOMS\n'
        conf += f'{self._N*2}\n'
        conf += 'ITEM: BOX BOUNDS pp pp pp\n'
        conf += f'{self._box[0]} {self._box[1]}\n'
        conf += f'{self._box[2]} {self._box[3]}\n'
        conf += f'{self._box[4]} {self._box[5]}\n'       
        conf += 'ITEM: ATOMS id mol type x y z c_quat[1] c_quat[2] c_quat[3] c_quat[4] shape[0] shape[1] shape[2]\n'
        
        return conf + ''.join([s.get_ovt_lammps_output() for s in self._strands])
    
    '''
    description: get lammps datafile of system
    param {*} self
    param {*} filename datafilename
    param {*} top_filename topfilename
    return {*}
    '''
    def get_lammps_data_file(self,filename = 'single.data',top_filename = 'topsingle.data',path = os.getcwd(),message=''):
        if len(self._nucleotide_to_strand) != self._N :
            self.map_nucleotides_to_strands()
        if len(self._nucleotides) != self._N :
            self.update_nucleotide_list()        
   
        # get total number of bonds
        N_bonds = 0
        for strand in self._strands:
            N_bonds += strand.get_lammps_N_of_bonds_strand()

        out_name = str(filename)
        out = open(os.path.join(path,out_name),"w")

        out.write('# message ' + message + '\n')
        out.write('# LAMMPS data file at time %d\n' % self._time)
        out.write('%d atoms\n' % self._N)
        out.write('%d ellipsoids\n' % self._N)
        out.write('%d bonds\n' % N_bonds)
        out.write('\n')
        out.write('8 atom types\n')
        out.write('1 bond types\n')
        out.write('\n')
        out.write('# System size\n')
        out.write('%f %f xlo xhi\n' % (self._box[0], self._box[1]))
        out.write('%f %f ylo yhi\n' % (self._box[2], self._box[3]))
        out.write('%f %f zlo zhi\n' % (self._box[4], self._box[5]))

        out.write('\n')
        out.write('Masses\n')
        out.write('\n')
        out.write('1 3.1575\n')
        out.write('2 3.1575\n')
        out.write('3 3.1575\n')
        out.write('4 3.1575\n')
        out.write('5 3.1575\n')
        out.write('6 3.1575\n')
        out.write('7 3.1575\n')
        out.write('8 3.1575\n')

        out.write('\n')
        out.write('# Atom-ID, type, position, molecule-ID, ellipsoid flag, density\n')
        out.write('Atoms\n')
        out.write('\n')

        for nucleotide in self._nucleotides:
            out.write('%d %d %.8f %.8f %.8f %d 1 1\n' \
                  % (nucleotide.index + 1, nucleotide._base + 1, \
                     nucleotide.cm_pos[0], nucleotide.cm_pos[1], nucleotide.cm_pos[2], \
                     self._nucleotide_to_strand[nucleotide.index] + 1))

        out.write('\n')
        out.write('# Atom-ID, translational, rotational velocity\n')
        out.write('Velocities\n')
        out.write('\n')

        for nucleotide in self._nucleotides:
            # v_rescaled = np.array(nucleotide._v) / np.sqrt(mass_in_lammps)
            # L_rescaled = np.array(nucleotide._L) * np.sqrt(inertia_in_lammps)
            out.write(f"{nucleotide.index + 1} 0 0 0 0 0 0\n")

        out.write('\n')
        out.write('# Atom-ID, shape, quaternion\n')
        out.write('Ellipsoids\n')
        out.write('\n')

        for nucleotide in self._nucleotides:
            out.write(\
            "%d %22.15le %22.15le %22.15le %.8f %.8f %.8f %.8f\n"  \
              % (nucleotide.index + 1, 1.1739845031423408, 1.1739845031423408, 1.1739845031423408, \
            nucleotide._quat[0], nucleotide._quat[1], nucleotide._quat[2], nucleotide._quat[3]))

        out.write('\n')
        out.write('# Bond topology\n')
        out.write('Bonds\n')
        out.write('\n')
        idx = 1
        for strand in self._strands:
            bonds = strand.get_lammps_bonds()
            for b in bonds:
                out.write("%d  %d  %s\n" % (idx, 1, b))
                idx += 1
        out.close()


        #top file
        out = open(os.path.join(path,str(top_filename)),'w')
        out.write('box_arr %f %f %f %f %f %f\n' % (self._box[0], self._box[1],self._box[2], self._box[3],self._box[4], self._box[5]))
        out.write('total_atoms %d\n' % (self._N))
        if self.probe_mol:
            out.write('probe_n %d\n' % (self._strands[self.probe_mol-1]._last+1))
            out.write('probe_mol %d\n' % self.probe_mol)
        out.write('#id strand base n5\n')
        for strand in self._strands:
            for n in strand._nucleotides:
                if strand._circular:
                    if n.index == strand._first:
                        n3 = strand._last
                    else:
                        n3 = n.index - 1
                    if n.index == strand._last:
                        n5 = strand._first
                    else:
                        n5 = n.index + 1
                else:
                    if n.index == strand._first:
                        n3 = -1
                    else:
                        n3 = n.index - 1
                    if n.index == strand._last:
                        n5 = -1
                    else:
                        n5 = n.index + 1
                out.write('%d %d %d %d\n' %(n.index,self._nucleotide_to_strand[n.index],n._base,n5))
        out.close()
        print('top data done!')
    
        print(f'Wrote data to {out_name}')



    lammps_output = property(get_lammps_output)
    ovt_lammps_output = property(get_ovt_lammps_output)
    lammps_data_file = property(get_lammps_data_file)


    #graph system
    def get_system_graph(self,path = None,filename = ''):
        if path:
            self.frame_path = path
        if len(self._nucleotides) != self._N:
            sys.exit(r'please use self.update_nucleotide_list updata self._nucleotides first\n' + f'{len(self._nucleotides)} {self._N}')
        plt.figure(4,figsize=(8*2,6), dpi=300)
        plt.subplot(121)
        # note that must use after self.get_hb_pair()
        nodesizemin = 50
        nodesizelen = 500
        G = nx.DiGraph()
        G.add_nodes_from(range(1,self._N_strands+1))
        
        #position
        probe_list = [x for x in range(1,self.probe_mol+1)]
        target_list = [x for x in range(self.probe_mol+1,self._N_strands+1)]
        pre_list = []
        pos_probe = dict([(x.index+1,self._nucleotides[x._first].pos_back_xy) for x in self._strands[:self.probe_mol]])
        pos_target = dict([(x.index+1,self._nucleotides[x._last].pos_back_xy) for x in self._strands[self.probe_mol:]])
        pos = {}
        pos.update(pos_probe)
        pos.update(pos_target)
        

        hb_percentage_list_probe = np.array([float(s.H_interaction_number/s._n) for s in self._strands[:self.probe_mol]])
        hb_mol_number_list_probe = [len(s.H_interactions) for s in self._strands[:self.probe_mol]]

        hb_percentage_list_target = np.array([float(s.H_interaction_number/s._n) for s in self._strands[self.probe_mol:]])
        hb_mol_number_list_target = [len(s.H_interactions) for s in self._strands[self.probe_mol:]]

        weight = {}
        # deal pre and add mirror
        pos_mirror = {}
        box = self.l_box[:2]
        for i in self._strands[:self.probe_mol]:
            i_pos =np.array(pos[i.index+1])
            for j in i.H_interactions:
                j_pos =np.array(pos[j+1])
                diff = np.rint((j_pos-i_pos)/ box)
                if np.all(diff == 0):
                    G.add_edge(j+1,i.index+1)
                else:
                    mirror_id = (i.index+1,diff[0],diff[1])
                    if mirror_id not in pos_mirror:
                        pos_mirror[mirror_id] = tuple(np.array(pos[i.index+1])+box*diff)
                    G.add_edge(j+1,mirror_id)
        pos.update(pos_mirror)

        #mirror set
        mirror_list = list(pos_mirror.keys())
        hb_percentage_list_mirror = np.array([hb_percentage_list_probe[x[0]-1] for x in mirror_list])
        hb_mol_number_list_mirror = [hb_mol_number_list_probe[x[0]-1] for x in mirror_list]
        #        weight[(j+1,i.index+1)] = float(i.H_interactions[j]) / i._n
        # nx.draw(G,nx.spring_layout(G),with_labels = True)
        
        nodes_probe = nx.draw_networkx_nodes(G,pos_probe,nodelist = probe_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_probe,node_color=hb_mol_number_list_probe,node_shape= 's',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'purple')
        nodes_target = nx.draw_networkx_nodes(G,pos_target,nodelist = target_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_target,node_color=hb_mol_number_list_target,node_shape= 'o',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'red')
        nodes_mirror =  nx.draw_networkx_nodes(G,pos_mirror,nodelist = mirror_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_mirror,node_color=hb_mol_number_list_mirror,node_shape= 's',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.2,edgecolors = 'purple')
        nodes_probe.set_zorder(-1)
        nodes_target.set_zorder(0)
        nodes_mirror.set_zorder(-2)
        arrow_style = mpl.patches.ArrowStyle.Fancy(head_length=.3, head_width=.3, tail_width=.05)
        edges = nx.draw_networkx_edges(G,pos,node_size=10,width=0.5,arrowstyle=arrow_style,arrowsize=8,alpha=0.9)#,edge_color=weight,edge_cmap='tab10',edge_vmin=0.,edge_vmax=1.)
        # nx.draw_networkx_labels(G,pos)
        
        plt.plot([self._box[0],self._box[1],self._box[1],self._box[0],self._box[0]],[self._box[2],self._box[2],self._box[3],self._box[3],self._box[2]])
        #nx.draw_networkx_edge_labels(G,pos,edge_labels = weight,label_pos = 0.5,font_size=16,font_weight='bold')    
        plt.xlim(self._box[0]-1./4*self.l_box[0],self._box[1]+1./4*self.l_box[0])
        plt.ylim(self._box[2]-1./4*self.l_box[1],self._box[3]+1./4*self.l_box[1])
        plt.gca().set_aspect("equal")


        plt.subplot(122)
        # note that must use after self.get_hb_pair()
        G = nx.DiGraph()
        G.add_nodes_from(range(1,self._N_strands+1))
        
        #position
        probe_list = [x for x in range(1,self.probe_mol+1)]
        target_list = [x for x in range(self.probe_mol+1,self._N_strands+1)]
        pre_list = []
        pos_probe = dict([(x.index+1,(self._nucleotides[x._last].cm_pos[0],self._nucleotides[x._last].cm_pos[1])) for x in self._strands[:self.probe_mol]])
        pos_target = dict([(x.index+1,(self._nucleotides[x._first].cm_pos[0],self._nucleotides[x._first].cm_pos[1])) for x in self._strands[self.probe_mol:]])

        nodes_probe = nx.draw_networkx_nodes(G,pos_probe,nodelist = probe_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_probe,node_color=hb_mol_number_list_probe,node_shape= 's',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'purple')
        nodes_target = nx.draw_networkx_nodes(G,pos_target,nodelist = target_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_target,node_color=hb_mol_number_list_target,node_shape= 'o',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'red')
        nodes_probe.set_zorder(-1)
        nodes_target.set_zorder(0)    
        nodes_probe_signle = nx.draw_networkx_nodes(G,pos_probe,nodelist = probe_list , node_size = nodesizemin,node_color='w',node_shape= '.',alpha = 0.8,edgecolors = 'purple')
        nodes_target_single = nx.draw_networkx_nodes(G,pos_target,nodelist = target_list , node_size = nodesizemin,node_color='w',node_shape= '.',alpha = 0.8,edgecolors = 'red')
        nodes_probe_signle.set_zorder(1)
        nodes_target_single.set_zorder(2)
        plt.colorbar(nodes_probe,orientation = 'vertical')
        plt.plot([self._box[0],self._box[1],self._box[1],self._box[0],self._box[0]],[self._box[2],self._box[2],self._box[3],self._box[3],self._box[2]])
        plt.xlim(self._box[0]-1./4*self.l_box[0],self._box[1]+1./4*self.l_box[0])
        plt.ylim(self._box[2]-1./4*self.l_box[1],self._box[3]+1./4*self.l_box[1])
        plt.gca().set_aspect("equal")

        path = os.path.join(self.frame_path,filename+f'_{self._time}'+'.jpg')
        plt.savefig(path,dpi = 300)
        plt.close()


    def __del__(self):
        Nucleotide.index = 0
        Strand.index = 0

    def get_system_last_graph(self,path = None,filename = ''):
        if path:
            self.frame_path = path
        if len(self._nucleotides) != self._N:
            sys.exit(r'please use self.update_nucleotide_list updata self._nucleotides first\n' + f'{len(self._nucleotides)} {self._N}')
        plt.figure(4,figsize=(8*2,6), dpi=300)
        plt.subplot(121)
        # note that must use after self.get_hb_pair()
        nodesizemin = 50
        nodesizelen = 500
        G = nx.DiGraph()
        G.add_nodes_from(range(1,self._N_strands+1))
        
        #position
        probe_list = [x for x in range(1,self.probe_mol+1)]
        target_list = [x for x in range(self.probe_mol+1,self._N_strands+1)]
        pre_list = []
        pos_probe = dict([(x.index+1,self._nucleotides[x._first].pos_back_xy) for x in self._strands[:self.probe_mol]])
        pos_target = dict([(x.index+1,self._nucleotides[x._last].pos_back_xy) for x in self._strands[self.probe_mol:]])
        pos = {}
        pos.update(pos_probe)
        pos.update(pos_target)
        

        hb_percentage_list_probe = np.array([float(s.H_interaction_number/s._n) for s in self._strands[:self.probe_mol]])
        hb_mol_number_list_probe = [len(s.H_interactions) for s in self._strands[:self.probe_mol]]

        hb_percentage_list_target = np.array([float(s.H_interaction_number/s._n) for s in self._strands[self.probe_mol:]])
        hb_mol_number_list_target = [len(s.H_interactions) for s in self._strands[self.probe_mol:]]

        weight = {}
        # deal pre and add mirror
        pos_mirror = {}
        box = self.l_box[:2]
        for i in self._strands[:self.probe_mol]:
            i_pos =np.array(pos[i.index+1])
            for j in i.H_interactions:
                j_pos =np.array(pos[j+1])
                diff = np.rint((j_pos-i_pos)/ box)
                if np.all(diff == 0):
                    G.add_edge(j+1,i.index+1)
                else:
                    mirror_id = (i.index+1,diff[0],diff[1])
                    if mirror_id not in pos_mirror:
                        pos_mirror[mirror_id] = tuple(np.array(pos[i.index+1])+box*diff)
                    G.add_edge(j+1,mirror_id)
        pos.update(pos_mirror)

        #mirror set
        mirror_list = list(pos_mirror.keys())
        hb_percentage_list_mirror = np.array([hb_percentage_list_probe[x[0]-1] for x in mirror_list])
        hb_mol_number_list_mirror = [hb_mol_number_list_probe[x[0]-1] for x in mirror_list]
        #        weight[(j+1,i.index+1)] = float(i.H_interactions[j]) / i._n
        # nx.draw(G,nx.spring_layout(G),with_labels = True)
        
        nodes_probe = nx.draw_networkx_nodes(G,pos_probe,nodelist = probe_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_probe,node_color=hb_mol_number_list_probe,node_shape= 's',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'purple')
        nodes_target = nx.draw_networkx_nodes(G,pos_target,nodelist = target_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_target,node_color=hb_mol_number_list_target,node_shape= 'o',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.5,edgecolors = 'red')
        nodes_mirror =  nx.draw_networkx_nodes(G,pos_mirror,nodelist = mirror_list , node_size = nodesizemin+nodesizelen*hb_percentage_list_mirror,node_color=hb_mol_number_list_mirror,node_shape= 's',cmap= 'rainbow',vmin = 0 ,vmax = 4,alpha = 0.2,edgecolors = 'purple')
        nodes_probe.set_zorder(-1)
        nodes_target.set_zorder(0)
        nodes_mirror.set_zorder(-2)
        arrow_style = mpl.patches.ArrowStyle.Fancy(head_length=.3, head_width=.3, tail_width=.05)
        edges = nx.draw_networkx_edges(G,pos,node_size=10,width=0.5,arrowstyle=arrow_style,arrowsize=8,alpha=0.9)#,edge_color=weight,edge_cmap='tab10',edge_vmin=0.,edge_vmax=1.)
        # nx.draw_networkx_labels(G,pos)
        
        plt.plot([self._box[0],self._box[1],self._box[1],self._box[0],self._box[0]],[self._box[2],self._box[2],self._box[3],self._box[3],self._box[2]])
        #nx.draw_networkx_edge_labels(G,pos,edge_labels = weight,label_pos = 0.5,font_size=16,font_weight='bold')    
        plt.xlim(self._box[0]-1./4*self.l_box[0],self._box[1]+1./4*self.l_box[0])
        plt.ylim(self._box[2]-1./4*self.l_box[1],self._box[3]+1./4*self.l_box[1])
        plt.gca().set_aspect("equal")
        plt.colorbar(nodes_probe,orientation = 'vertical')
        path = os.path.join(self.frame_path,filename+f'_{self._time}_single'+'.jpg')
        plt.savefig(path,dpi = 300)
        plt.close()

    #get system slice
    def get_slice_system(self,probe_mol_id=[],target_mol_id=[]):
        s = System(np.array(self._box),time=self._time)
        Nucleotide.index=0
        Strand.index=0
        for i in probe_mol_id:
            s.add_strands(self._strands[i].copy())
        for i in target_mol_id:
            s.add_strands(self._strands[i].copy())
        s.update_probe_target_system(probe_mol=len(probe_mol_id))
        return s

    def get_top_file(self,top_filename = 'topsingle.data',path = os.getcwd()):
        #top file
        if len(self._nucleotide_to_strand) != self._N :
            self.map_nucleotides_to_strands()
        if len(self._nucleotides) != self._N :
            self.update_nucleotide_list()        
        out = open(os.path.join(path,str(top_filename)),'w')
        out.write('box_arr %f %f %f %f %f %f\n' % (self._box[0], self._box[1],self._box[2], self._box[3],self._box[4], self._box[5]))
        out.write('total_atoms %d\n' % (self._N))
        if self.probe_mol:
            out.write('probe_n %d\n' % (self._strands[self.probe_mol-1]._last+1))
            out.write('probe_mol %d\n' % self.probe_mol)
        out.write('#id strand base n5\n')
        for strand in self._strands:
            for n in strand._nucleotides:
                if strand._circular:
                    if n.index == strand._first:
                        n3 = strand._last
                    else:
                        n3 = n.index - 1
                    if n.index == strand._last:
                        n5 = strand._first
                    else:
                        n5 = n.index + 1
                else:
                    if n.index == strand._first:
                        n3 = -1
                    else:
                        n3 = n.index - 1
                    if n.index == strand._last:
                        n5 = -1
                    else:
                        n5 = n.index + 1
                out.write('%d %d %d %d\n' %(n.index,self._nucleotide_to_strand[n.index],n._base,n5))
        out.close()
        print('top data done!')



#funtions of plot
class Plot_system():

    def __init__(self) -> None:
        self.n_probe = None
        self.n_target = None

    def plot_system(self,file_name):
        d = pd.read_csv(file_name,sep = ' ',header=None)
        plt.figure(4,figsize=(8,6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplot(1,1,1)
        plt.plot(d.iloc[:,0],d.iloc[:,-1])
        plt.savefig(file_name + '.jpg',dpi=300)
        plt.close()
    
    def plot_probe_system(self,file_name,v_max=1):
        d = np.array(pd.read_csv(file_name,sep = ' ',header=None))
        plt.figure(4,figsize=(8*2,6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplot(1,2,1)
        # bins = np.linspace(1,self.n_probe,self.n_probe)
        # print(bins)
        # nbin = len(bins)-1
        # cmap = mpl.cm.get_cmap('rainbow',nbin) 
        # norm = mpl.colors.BoundaryNorm(bins,nbin)
        # im = mpl.cm.ScalarMappable(norm = norm , cmap = cmap)
        # plt.colorbar(im,orientation = 'vertical')
        # for i in range(self.n_probe):
        #     plt.plot(d[:,0],d[:,(i+1)],alpha = 0.7,color = cmap(i/nbin) , label = f'{i+1}')

        x_n,y_n = d.transpose()[1:(self.n_probe+1),:].shape
        y = np.linspace(1,x_n,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d.transpose()[1:(self.n_probe+1),:], cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[1:(self.n_probe+1),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')

        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(0.5,self.n_probe+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(1,self.n_probe,6))
        # plt.colorbar(im,orientation = 'vertical')
        # plt.legend(bbox_to_anchor=(1.05, 0),loc=3, borderaxespad=0,fontsize = 7.5)
        plt.subplot(1,2,2)
        x_n,y_n =  d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:].shape
        y = np.linspace(1,x_n,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:], cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')
        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(0.5,self.n_target+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(1,self.n_probe,6))
        plt.colorbar(im,orientation = 'vertical')
        
        plt.savefig(file_name + 'probe.jpg',dpi=300)

        plt.close()
        pass

    # for many files
    
    @classmethod
    def hb_plots(self,file_names,out_file_name = 'hb_ave.png',singleflag = False,label = []):

        if isinstance(file_names,str):
            file_names = [file_names]
        hb_num = []
        file_n = len(file_names)
        print(f'will deal with {file_n} files ! \n')
        if singleflag:
            if len(label) != file_n:
                exit('len of label not correct!')
            for f in file_names:
                try:
                    hb_num.append(np.array(pd.read_csv(f,sep = ' ',header=None),np.float64))
                except:
                    exit('cannot find the file')
            

                #base setting

            font = 'Arial'
            font_size = 22
            timestep = 0.005
            y_shift_arg = 25
            lw = 2 # linewidth
            colors = plt.get_cmap('rainbow')(np.linspace(0,1,file_n))
            

            #plot hb_number
            plt.figure(4,figsize=(12*2,6*2), dpi=300)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            ###########################################################################################
            x = hb_num[0][:,0]/100000
            
            plt.subplot(231)
            for i,c,l in zip(hb_num,colors,label):
                plt.plot(x,i[:,1],color=c,linewidth =lw,label=l)
            time_root = timestep * 100000
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('hb percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #    plt.legend(loc='upper right', prop={'family':font, 'size':font_size})
            ###########################################################################################
            plt.subplot(232)
            for i,c,l in zip(hb_num,colors,label):
                plt.plot(x,i[:,2],color = c,linewidth =lw,label=l)

            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('probe percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(233)
            for i,c,l in zip(hb_num,colors,label):
                plt.plot(x,i[:,3],color = c,linewidth =lw,label=l)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('target percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(235)
            for i,c,l in zip(hb_num,colors,label):
                plt.plot(x,i[:,4],color = c,linewidth =lw,label=l)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('probe single percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(236)
            for i,c,l in zip(hb_num,colors,label):
                plt.plot(x,i[:,5],color = c,linewidth =lw,label=l)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('target single percentage', fontdict={'family' : font, 'size':font_size})
            plt.legend(loc='lower right', prop={'family':font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)


        # ########### save figure

            plt.savefig(out_file_name,dpi=300)#, bbox_inches="tight")
        # #   plt.show()
        #     print(str(hb_num_file0)+'.png  #done')
        #     #plot moule number

            plt.close('all')      
        else:

            for f in file_names:
                try:
                    hb_num.append(np.array(pd.read_csv(f,sep = ' ',header=None),np.float64))
                except:
                    exit('cannot find the file')
            
            hb_ave = hb_num[0]
            if file_n > 1 :
                for i in range(1,file_n):
                    hb_ave += hb_num[i]
                hb_ave /= file_n

                #base setting
            np.savetxt('hb_ave.txt',hb_ave,fmt='%.7f',delimiter = ' ')

            font = 'Arial'
            font_size = 22
            timestep = 0.005
            y_shift_arg = 25
            lw = 2 # linewidth
            #plot hb_number
            plt.figure(4,figsize=(12*2,6*2), dpi=300)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            ###########################################################################################
            x = hb_num[0][:,0]/100000
            plt.subplot(231)
            for i in hb_num:
                plt.plot(x,i[:,1],'g-',linewidth =lw)
            plt.plot(x,hb_ave[:,1],'r-',linewidth =lw)
            time_root = timestep * 100000
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('hb percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #    plt.legend(loc='upper right', prop={'family':font, 'size':font_size})
            ###########################################################################################
            plt.subplot(232)
            for i in hb_num:
                plt.plot(x,i[:,2],'g-',linewidth =lw)
            plt.plot(x,hb_ave[:,2],'r-',linewidth =lw)

            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('probe percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(233)
            for i in hb_num:
                plt.plot(x,i[:,3],'g-',linewidth =lw)
            plt.plot(x,hb_ave[:,3],'r-',linewidth =lw)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('target percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(235)
            for i in hb_num:
                plt.plot(x,i[:,4],'g-',linewidth =lw)
            plt.plot(x,hb_ave[:,4],'r-',linewidth =lw)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('probe single percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)
        #     ###########################################################################################
            plt.subplot(236)
            for i in hb_num:
                plt.plot(x,i[:,5],'g-',linewidth =lw)
            plt.plot(x,hb_ave[:,5],'r-',linewidth =lw)
            time_root = timestep * 100000
            #plt.scatter(lk.iloc[:,0]/10000,lk.loc[:,3],c= 'r',s=0.1,label='lk') 
            right = x[-1]
            down = 0
            up = 1
            y_shift = (up-down)/y_shift_arg

            plt.ylim(down-y_shift,up+y_shift)
            plt.xlim(-100,right)
            plt.xticks(np.linspace(0,right,6,endpoint=True),fontproperties = font, fontsize=font_size)
            plt.yticks(np.linspace(down,up,6,endpoint=True),fontproperties = font, fontsize=font_size)

            plt.xlabel('t*'+str(time_root)+chr(964), fontdict={'family' : font, 'size':font_size})
            plt.ylabel('target single percentage', fontdict={'family' : font, 'size':font_size})
            plt.grid()
            plt.margins(x=0)


        # ########### save figure

            plt.savefig(out_file_name,dpi=300)#, bbox_inches="tight")
        # #   plt.show()
        #     print(str(hb_num_file0)+'.png  #done')
        #     #plot moule number

            plt.close('all')      

    @classmethod
    def kde_plots(self,file_names,out_file_name = 'kde_ave.png',singleflag = False,label = []):
        from scipy.stats import gaussian_kde

        #filesystem
        if isinstance(file_names,str):
            file_names = [file_names]
        file_n = len(file_names)

        print(f'will deal with {file_n} files ! \n')
        #plot system

        font = 'Arial'
        font_size = 22
        timestep = 0.005
        y_shift_arg = 25
        lw = 2 # linewidth
        xmin = 0
        xmax = 20
        ymin = 0
        ymax = 0.3

        #plot hb_number

        plt.figure(4,figsize=(8,6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplots_adjust(left=0.14,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.2)
        x = np.linspace(xmin,xmax,2000)

        
        if singleflag:
            colors = plt.get_cmap('rainbow')(np.linspace(0,1,file_n))
            for f,c,l in zip(file_names,colors,label):
                try:
                    datas = (np.array(pd.read_csv(f,header=None,sep='\s+'),dtype=np.float64).ravel().tolist())
                    
                    kde = gaussian_kde(datas)
                    plt.plot(x,kde(x),linewidth =lw,color = c, label = l)
                except:
                    exit('cannot find the file')
                
                
            plt.legend()
        else:
            datas = []
            for f in file_names:
                try:
                    datas += np.array(pd.read_csv(f,header=None,sep='\s+'),dtype=np.float64).ravel().tolist()[1:]
                except:
                    exit('cannot find the file')
            kde = gaussian_kde(datas)
            plt.plot(x,kde(x),linewidth =lw)
            np.savetxt('kde_ave.txt',np.array(datas),fmt='%.6f',delimiter = ' ')

            
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.xticks(np.linspace(xmin,xmax,5),fontproperties = font, fontsize=font_size)
        plt.yticks(np.linspace(ymin,ymax,6,endpoint=True),fontproperties = font, fontsize=font_size)
        plt.xlabel(chr(963), fontdict={'family' : font, 'size':font_size})
        plt.ylabel('Probability', fontdict={'family' : font, 'size':font_size})
        plt.margins(x=0)
        plt.savefig(out_file_name,dpi=300)
        plt.close()
        print('kde single plot done!')

    @classmethod
    def hb_mol_kde_plots(self,file_names,out_file_name = 'hb_kde_ave.png',singleflag = False,label = [],probe_mol = None):
        from scipy.stats import gaussian_kde

        #filesystem
        if isinstance(file_names,str):
            file_names = [file_names]
        file_n = len(file_names)
        if not probe_mol:
            probe_mol = self.n_probe
        if not isinstance(probe_mol,int):
            exit('probe_mol is not a int type!! please set int probe_mol')
        print(f'probe_mol is {probe_mol} ! \n')
        print(f'will deal with {file_n} files ! \n')
        #plot system

        font = 'Arial'
        font_size = 22
        timestep = 0.005
        y_shift_arg = 25
        lw = 2 # linewidth
        xmin = -1
        xmax = 5
        ymin = 0
        ymax = 1

        #plot hb_number

        plt.figure(4,figsize=(8,6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplots_adjust(left=0.14,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.2)
        x = np.linspace(xmin,xmax,7,endpoint=True)

        
        if singleflag:
            colors = plt.get_cmap('rainbow')(np.linspace(0,1,file_n))
            for f,c,l in zip(file_names,colors,label):
                try:
                    datas = (np.array(pd.read_csv(f,header=None,sep='\s+'),dtype=np.float64).ravel().tolist())
                    
                    kde = gaussian_kde(datas)
                    plt.plot(x,kde(x),linewidth =lw,color = c, label = l)
                except:
                    exit('cannot find the file')
                
                
            plt.legend()
        else:
            datas = []
            for f in file_names:
                try:
                    datas += np.array(pd.read_csv(f,header=None,sep='\s+'),dtype=int)[-500:,(probe_mol+1):].ravel().tolist()
                except:
                    exit('cannot find the file')
            import collections
            c = collections.Counter(datas)
            # kde = gaussian_kde(datas)
            # plt.plot(x,kde(x),linewidth =lw)
            x = [0,1,2,3,4,5]
            y = np.zeros((6,),dtype=np.int64)
            for i in x:
                y[i] = c[i]
            # plt.plot(x,y,linewidth =lw)
            plt.hist(datas,bins=6,range=(-0.5,5.5),density=True,color='r',align='mid',rwidth=0.5)
            np.savetxt('hb_kde_count.txt',np.array(y),fmt='%d',delimiter = ' ',newline=' ')
            np.savetxt('hb_kde_ave.txt',np.array(datas),fmt='%d',delimiter = ' ',newline=' ')

            
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.xticks(np.linspace(xmin,xmax,7),[ '' if i in [xmin,xmax] else str(i) for i in np.linspace(xmin,xmax,7) ],fontproperties = font, fontsize=font_size)
        plt.yticks(np.linspace(ymin,ymax,6,endpoint=True),fontproperties = font, fontsize=font_size)
        plt.xlabel('N', fontdict={'family' : font, 'size':font_size})
        plt.ylabel('Probability', fontdict={'family' : font, 'size':font_size})
        plt.margins(x=0)
        plt.savefig(out_file_name,dpi=300)
        plt.close()
        print('hb kde single plot done!')

    @classmethod
    def plot_nick_system(self,file_name,v_max=1):
        d = np.array(pd.read_csv(file_name,sep = ' ',header=None))
        plt.figure(4,figsize=(8,3), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.subplot(1,2,1)
        n_probe = d.shape[1] -1 

        d1 = d[:,1:(n_probe//2+1)]
        d2 = d[:,(n_probe//2+1):]
        x_n,y_n = d1.transpose().shape
        y = np.linspace(1,x_n+1,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d1.transpose(), cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[1:(self.n_probe+1),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')
        x_n,y_n = d2.transpose().shape
        y = np.linspace(-x_n-1,-1,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d2.transpose(), cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(-n_probe//2-0.5,n_probe//2+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(-n_probe//2,n_probe//2,5))



        plt.subplot(1,2,2)
        # bins = np.linspace(1,self.n_probe,self.n_probe)
        # print(bins)
        # nbin = len(bins)-1
        # cmap = mpl.cm.get_cmap('rainbow',nbin) 
        # norm = mpl.colors.BoundaryNorm(bins,nbin)
        # im = mpl.cm.ScalarMappable(norm = norm , cmap = cmap)
        # plt.colorbar(im,orientation = 'vertical')
        # for i in range(self.n_probe):
        #     plt.plot(d[:,0],d[:,(i+1)],alpha = 0.7,color = cmap(i/nbin) , label = f'{i+1}')
        d1 = d[:,1:11]
        d2 = d[:,(n_probe-9):(n_probe+1)]
        x_n,y_n = d1.transpose().shape
        y = np.linspace(1,x_n+1,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d1.transpose(), cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[1:(self.n_probe+1),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')
        x_n,y_n = d2.transpose().shape
        y = np.linspace(-x_n-1,-1,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d2.transpose(), cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(-10-0.5,10+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(-10,10,5))
        # plt.colorbar(im,orientation = 'vertical')
        # plt.legend(bbox_to_anchor=(1.05, 0),loc=3, borderaxespad=0,fontsize = 7.5)
        # plt.colorbar(im,orientation = 'vertical')
        print(f'total {n_probe} bp!')
        plt.savefig(file_name + '.jpg',dpi=300)
        plt.close()

    @classmethod
    def plot_lk(self,filename,top=32):
        plt.figure(4,figsize=(8, 9), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    ###########################################################################################
        plt.subplots_adjust(left=0.1,bottom=0.08,right=0.95,top=0.95,wspace=0.3,hspace=0.2)
        front_size = 18
        lk = pd.read_csv(filename,sep = ' ',header=None)
    
        x = np.array(lk.iloc[:,0]/100000)
        y1 = np.array(lk.iloc[:,1])
        y2 = np.array(lk.iloc[:,2])
        y3 = np.array(lk.iloc[:,3])

    
        font = {'family':'Arial','weight':'normal','size':24}
        plt.subplot(311)
        #plt.title('Twist-dominated',fontsize=2*front_size)
    
        plt.plot(x[0:4000],y1[0:4000],'g-',linewidth=1,label='Tw')
        #plt.xlabel('t*'+str(100)+chr(964),fontsize=front_size)
        plt.ylabel('Tw',font,labelpad=0)
        plt.xlim(0,4000)   
        plt.xticks(range(0,4100,1000),['','','','',''],fontproperties = 'Arial', fontsize=front_size)
        lkdown = top-3
        lkup = top+2
        plt.ylim(lkdown,lkup)
        plt.yticks(range(lkdown,lkup+1),fontproperties = 'Arial', fontsize=front_size)
    
        plt.margins(x=0)
        # plt.grid()
    
        plt.subplot(312)
        plt.plot(x[0:4000],y2[0:4000],'b-',linewidth=1,label='Wr')
        #plt.xlabel('t*'+str(100)+chr(964),fontsize=front_size)
        plt.ylabel('Wr',font,labelpad=0)
        plt.xlim(0,4000)
        plt.xticks(range(0,4100,1000),['','','','',''],fontproperties = 'Arial', fontsize=front_size)
        lkdown = top-33
        lkup = top-28
        plt.ylim(lkdown,lkup)
        plt.yticks(range(lkdown,lkup+1),fontproperties = 'Arial', fontsize=front_size)
    
        plt.margins(x=0)

        plt.subplot(313)
        plt.plot(x[0:4000],y3[0:4000],'y-',linewidth=1,label='Lk',zorder = 2)
        plt.xlabel('t '+'(100'+chr(964)+')',font,labelpad=0)
        plt.ylabel('Lk',font,labelpad=0)
        plt.xlim(0,4000)
        plt.xticks(range(0,4100,1000),fontproperties = 'Arial', fontsize=front_size)
        lkdown = top-1
        lkup = top+4
        plt.ylim(lkdown,lkup)
        plt.yticks(range(lkdown,lkup+1),fontproperties = 'Arial', fontsize=front_size)
    
        plt.margins(x=0)
    
        plt.savefig(str(filename)+'.jpg',dpi=300)#, bbox_inches="tight")
        plt.close()
    
        print('lk_plot done')

    @classmethod
    def plot_contactmap(self,data,out_filename = 'contactmap'):
        plt.figure(4,figsize=(8, 4), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    ###########################################################################################
        plt.subplots_adjust(left=0.1,bottom=0.15,right=0.58,top=0.91,wspace=0.3,hspace=0.2)
        fonttitle = {'family':'Arial','weight':'normal','size':24}
        plt.subplot(111)
        n,m = data.shape
        fig = plt.imshow(data, cmap='rainbow', origin='lower',vmax = 8.0,vmin = 0.0)
        plt.xlabel('Base pair id',fonttitle,labelpad=0)
        plt.ylabel('Base pair id',fonttitle,labelpad=0)
        plt.xlim(0,n-1)   
        plt.ylim(0,n-1)   
        plt.xticks(np.linspace(0,m-1,6),np.linspace(0,m-1,6,dtype=int)+1)
        plt.yticks(np.linspace(0,n-1,6),np.linspace(0,n-1,6,dtype=int)+1)
        clb = plt.colorbar(fig,orientation = 'vertical')
        clb.set_ticks(np.linspace(0,8,9,dtype=int))
        clb.set_ticklabels(np.linspace(0,8,9,dtype=int))
        clb.set_label("Distance (" + chr(963) + ')',fontdict=fonttitle,rotation=270,labelpad=25)
        clb.minorticks_on()

        plt.savefig(out_filename+'.tif',dpi=300)
        plt.clf() 
        plt.close('all')

    @classmethod
    def plot_local_twist(self,filename):
        plt.figure(4,figsize=(8, 6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    ###########################################################################################
        plt.subplots_adjust(left=0.1,bottom=0.15,right=0.9,top=0.91,wspace=0.3,hspace=0.2)
        fonttitle = {'family':'Arial','weight':'normal','size':24}
        plt.subplot(111)
        data = np.array(pd.read_csv(filename,header= None,sep=' '))
        y,x= data.shape
        data = scipy.ndimage.gaussian_filter(data[:,1:],3,mode = 'wrap')

        xx = np.linspace(0,y-1,y)
        yy = np.linspace(1,x-1,x-1)

        xx,yy=np.meshgrid(xx,yy)
        w = 32*360/10.5/32
        im = plt.pcolormesh(xx,yy, data.transpose()-w, cmap='rainbow',vmin=-1,vmax=1, shading='nearest')

        plt.xlim(-0.5,y-0.5)
        plt.ylim(0.5,x-0.5)
        plt.xticks(np.linspace(0,y-1,3),['0','4000','8000'])
        plt.xlabel('t '+'(100'+chr(964)+')',fonttitle,labelpad=0)
        plt.yticks(np.linspace(0.5,x-0.5,6),np.linspace(1,x,6,dtype=int))

        plt.ylabel('Relative bp id',fonttitle,labelpad=0)
        clb = plt.colorbar(im,orientation = 'vertical')
        clb.set_ticks(np.linspace(-1,1,9,dtype=int))
        clb.set_label(f'${chr(937)}_3$' + r'(°/bp)',fontdict=fonttitle,rotation=270,labelpad=25)


        plt.savefig(filename+'.jpg',dpi=300)
        plt.close()

    @classmethod
    def plot_band_area(self,filename):
        fonttitle = {'family':'Arial','weight':'normal','size':24}
        plt.figure(4,figsize=(8, 6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    ###########################################################################################
        plt.subplots_adjust(left=0.1,bottom=0.15,right=0.9,top=0.91,wspace=0.3,hspace=0.2)    
        data = np.array(pd.read_csv(filename,header= None,sep=' '))
        y,x= data.shape
        data = data[:,1:]
        clist=['lightcyan','lightcoral']

        c = mpl.colors.LinearSegmentedColormap.from_list('qyp',clist)
        xx = np.linspace(0,y-1,y)
        yy = np.linspace(1,x-1,x-1)
        
        xx,yy=np.meshgrid(xx,yy)
        
        im = plt.pcolormesh(xx,yy,data.T, cmap=c,vmin=0,vmax=1, shading='nearest')

        plt.xlim(-0.5,y-0.5)
        plt.ylim(0.5,x-0.5)
        plt.xticks(np.linspace(0,y-1,3),['0','4000','8000'])
        plt.xlabel('t '+'(100'+chr(964)+')',fonttitle,labelpad=0)
        plt.yticks(np.linspace(1,x-1,6,dtype=int))

        plt.ylabel('Relative bp id',fonttitle,labelpad=0)


        plt.savefig(filename+'.jpg',dpi=300)
        plt.close()

class File_system(Plot_system):
    def __init__(self,pwd = os.getcwd() , filename = 'single.data'):
        self.n_probe = None
        self.n_target = None
        self._pwd = pwd
        self._filename = filename
        self._ovt_file = False
        self._sor_file = False
        self._rg_file = False
        self._hb_total_energy_file = False
        self._hb_file = False
        self._hb_single_file = False
        self._hb_single_mol_file = False
        self._cpu_time = False
        self._lk_file = False
        self._kde_file = False
        self._nick_file = False
        self._local_twist = False
        self._lk = 32
        self._band_file = False
        self._band_percentage_flie = False
        self._array_structure_file = False
        

        
    def plot_probe_system(self,file_name,v_max=1):
        d = np.array(pd.read_csv(file_name,sep = ' ',header=None))
        plt.figure(4,figsize=(8*2,6), dpi=300)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        cmap = mpl.colors.ListedColormap(['g','r','b'])     

        plt.subplot(1,2,1)



        x_n,y_n = d.transpose()[1:(self.n_probe+1),:].shape
        y = np.linspace(1,x_n,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d.transpose()[1:(self.n_probe+1),:], cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[1:(self.n_probe+1),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')

        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(0.5,self.n_probe+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(1,self.n_probe,6))
        # plt.colorbar(im,orientation = 'vertical')
        # plt.legend(bbox_to_anchor=(1.05, 0),loc=3, borderaxespad=0,fontsize = 7.5)
        plt.subplot(1,2,2)
        x_n,y_n =  d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:].shape
        y = np.linspace(1,x_n,x_n)
        x = np.linspace(0,y_n-1,y_n)
        xx,yy=np.meshgrid(x,y)
        im = plt.pcolormesh(xx,yy, d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:], cmap='rainbow',vmin=0,vmax=v_max, shading='nearest')
    #    im = plt.imshow(d.transpose()[(1+self.n_probe):(1+self.n_probe+self.n_target),:],aspect = 'auto',origin='lower',vmin=0,vmax=v_max,cmap='rainbow')
        plt.xlim(0-0.5,y_n-0.5)
        plt.ylim(0.5,self.n_target+0.5)
        plt.xticks(np.linspace(0,y_n-1,5))
        plt.yticks(np.linspace(1,self.n_target,6))
        plt.colorbar(im,orientation = 'vertical')
        
        plt.savefig(file_name + 'probe.jpg',dpi=300)

        plt.close()
        pass

    def add_sor_message(self,time : int , message : list):
        if not self._sor_file:
            self._sor_file = open(os.path.join(self._pwd,str(self._filename)+'_sor.data'),'w+')
            
        self._sor_file.write(f'{time} '+' '.join(str(x) for x in message)+'\n')

    def add_rg_message(self,time : int , message):
        if not self._rg_file:
            self._rg_file = open(os.path.join(self._pwd,str(self._filename)+'_rg.data'),'w+')
            
        self._rg_file.write(f'{time} '+' '.join(str(round(np.sqrt(sum(x)),6)) for x in message)+'\n')

    def add_hb_energy(self,time : int , message):
        if not self._hb_total_energy_file:
            self._hb_total_energy_file = open(os.path.join(self._pwd,str(self._filename)+'_hb_total_energy.data'),'w+')
        self._hb_total_energy_file.write(f'{time} ' + str(round(message,6)) + '\n')

    def add_hb_message(self,time : int , message):
        if not self._hb_file:
            self._hb_file = open(os.path.join(self._pwd,str(self._filename)+'_hb.data'),'w+')
            self._hb_single_file = open(os.path.join(self._pwd,str(self._filename)+'_hb_single.data'),'w+')
            self._hb_single_mol_file = open(os.path.join(self._pwd,str(self._filename)+'_hb_single_mol.data'),'w+')
        self._hb_file.write(f'{time} ' + ' '.join(str(round(x,6)) for x in message[0]) + '\n')
        self._hb_single_file.write(f'{time} ' + ' '.join(str(round(x,6)) for x in message[1]) + '\n')
        self._hb_single_mol_file.write(f'{time} ' + ' '.join(str(x) for x in message[2]) + '\n')
    
    def add_ovt_output(self, message: str):
        if not self._ovt_file:
            self._ovt_file = open(os.path.join(self._pwd,'ovt_' + str(self._filename)),'w+')
        self._ovt_file.write(message)

    def cpu_time(self,time:int,cpu_time:float):
        if not self._cpu_time:
            self._cpu_time = open(os.path.join(self._pwd,str(self._filename)) + 'cpu_time.data','w+')
        self._cpu_time.write(f'{time} {cpu_time:.3f}\n')

    def add_lk_message(self,time : int ,message : list):
        if not self._lk_file:
            self._lk_file = open(os.path.join(self._pwd,str(self._filename)) + '_top.data','w+')
        self._lk_file.write(f'{time} '+' '.join(str(round(x,6)) for x in message)+'\n')

    def add_local_twist(self,time:int, message: list):
        if not self._local_twist:
            self._local_twist = open(os.path.join(self._pwd,str(self._filename)) + '_local_twist.data','w+')
        self._local_twist.write(f'{time} '+' '.join(str(round(x,3)) for x in message)+'\n')

    def add_kde_message(self,time : int ,message : list):
        if not self._kde_file:
            self._kde_file = open(os.path.join(self._pwd,str(self._filename)) + '_kde.data','w+')
        self._kde_file.write(f'{time} '+' '.join(str(round(x,6)) for x in message)+'\n')

    def add_band_area(self,time : int ,message ):
        if not self._band_file:
            self._band_file = open(os.path.join(self._pwd,str(self._filename)) + '_band.data','w+')
        if not self._band_percentage_flie:
            self._band_percentage_flie = open(os.path.join(self._pwd,str(self._filename)) + '_band_percentage.data','w+')
        self._band_file.write(f'{time} '+' '.join(str(i) for i in message) + '\n')
        self._band_percentage_flie.write(f'{time} {np.sum(message)}\n')

    def add_nickheat(self,time,message):
        if not self._nick_file:
            self._nick_file = open(os.path.join(self._pwd,str(self._filename)) + '_nick.data','w+')
        self._nick_file.write(f'{time} '+' '.join(str(x) for x in message)+'\n')

    def add_array_structure(self,time : int ,message):
        if not self._array_structure_file:
            self._array_structure_file = open(os.path.join(self._pwd,str(self._filename)) + '_array_structure.data','w+')
        self._array_structure_file.write(f'{time} '+' '.join(str(x) for x in message)+'\n')
    
    # def string_file(self,filename,message):
    #     with open(filename,'w+') as f:
    #         f.write(message)
    def plot_try(self):
        try:
            self.plot_probe_system(os.path.join(self._pwd,str(self._filename)+'_hb_single_mol.data'),v_max=4)
        except:
            print('cannot plot hb number!')




    def __del__(self):
        if self._sor_file:
            self._sor_file.close()
            try:
                self.plot_system(os.path.join(self._pwd,str(self._filename)) + '_sor.data')
            except:
                print('cannot plot sor!')
            try:
                self.plot_probe_system(os.path.join(self._pwd,str(self._filename)) + '_sor.data')
            except:
                print('cannot plot probe sor')
            print('sor model done!')
        
        if self._rg_file:
            self._rg_file.close()
            print('rg model done!')
        
        if self._hb_total_energy_file:
            self._hb_total_energy_file.close()
            try:
                self.plot_system(os.path.join(self._pwd,str(self._filename)+'_hb_total_energy.data'))
            except:
                print('_hb_total_energy_file!')

        if self._hb_file:
            self._hb_file.close()
            self._hb_single_file.close()
            self._hb_single_mol_file.close()

            try:
                self.plot_probe_system(os.path.join(self._pwd,str(self._filename)+'_hb_single.data'))
            except:
                print('cannot plot hb single!')
            try:
                self.plot_probe_system(os.path.join(self._pwd,str(self._filename)+'_hb_single_mol.data'),v_max=4)
            except:
                print('cannot plot hb number!')
            print('hb model done!')
        
        if self._ovt_file:
            self._ovt_file.close()
            print('ovt mode done!')
        
        if self._cpu_time:
            self._cpu_time.close()
            try:
                self.plot_system(os.path.join(self._pwd,str(self._filename)) + 'cpu_time.data')
            except:
                print('cannot plot cpu_time')
            print('cpu_time done!')

        if self._lk_file:
            self._lk_file.close()    
            try:
                self.plot_lk(os.path.join(self._pwd,str(self._filename)) + '_top.data',top=self._lk)
            except:
                print('cannot plot _top')
            print('TW,WR,LK done!')
        
        if self._local_twist:
            self._local_twist.close()
            try:
                self.plot_local_twist(os.path.join(self._pwd,str(self._filename)) + '_local_twist.data')
            except:
                print('cannot plot _local_twist')
            print('local_twist done')


        if self._kde_file:
            self._kde_file.close()
            # try:
            #     self.kde_plots(os.path.join(self._pwd,str(self._filename)) + '_kde.data')
            # except:
            #     print('cannot plot kde')
            print('kde done!')
        
        if self._nick_file:
            self._nick_file.close()
            try:
                self.plot_nick_system(os.path.join(self._pwd,str(self._filename)) + '_nick.data')
            except:
                print('cannot plot _nick_system')
            print('nick file done!')

        if self._band_file:
            self._band_file.close()
            try:
                self.plot_band_area(os.path.join(self._pwd,str(self._filename)) + '_band.data')
            except:
                print('cannot plot _band!')
            print('band file done!')

        if self._band_percentage_flie:
            self._band_percentage_flie.close()
            try:
                self.plot_system(os.path.join(self._pwd,str(self._filename)) + '_band_percentage.data')
            except:
                print('cannot plot _band_per!')
            print('_band_percentage file done!')
        
        if self._array_structure_file:
            self._array_structure_file.close()


            print('_array_structure file done!')
# for text
if __name__ == "__main__":
    flag = 0# True for every fold else for single file
    num_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    filename_hb = [f'original_T_300_{i}.lammpstrj_hb.data' for i in num_names ]
    filename_kde = [f'original_T_300_{i}.lammpstrj_kde.data' for i in num_names ]
    filename_hb_mol = [f'original_T_300_{i}.lammpstrj_hb_single_mol.data' for i in num_names]
    if flag :
        import dumpreader as dr

        # for f in filename:
        #     w = File_system(pwd,f)
        #     w.plot_try()
        #     print(f + ' done!')
        #     del w
        para_dict = dr.Lammps_dumpreader.import_base_parameters('top.data')
        probe_mol = None
        if 'probe_mol' in para_dict:
            probe_mol = int(para_dict['probe_mol'])

        Plot_system.hb_mol_kde_plots(file_names=filename_hb_mol,out_file_name='hb_kde_ave.png',singleflag=False,probe_mol=probe_mol)
        Plot_system.hb_plots(file_names=filename_hb,out_file_name='ave.png')
        Plot_system.kde_plots(file_names=filename_kde,singleflag=False)

    else:
        r0 = [900,1500,3000,6000]
        r1 = [901,1501,3001,6001]#,
        r2 = [902,1502,3002,6002]
        r3 = [300220,300210,3002,300203]
        l0 = ['3-aa','5-aa','10-aa','20-aa']
        l1 = ['3-ac','5-ac','10-ac','20-ac']#,
        l2 = ['3-cc','5-cc','10-cc','20-cc']
        l3 = ['3-ccp','5-ccp','10-ccp','20-ccp']

        r = r3
        l = l3
        h = [f'hb{m}.txt' for m in r]
        k = [f'kde{m}.txt' for m in r]
        Plot_system.hb_plots(file_names=h,out_file_name='hb_main.png',singleflag=True,label=l)
        Plot_system.kde_plots(file_names=k,out_file_name='kde_main.png',singleflag=True,label=l)
    
