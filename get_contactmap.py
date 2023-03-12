'''
Author: qyp422
Date: 2023-02-11 23:48:45
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-02-13 17:35:23
Description: 

Copyright (c) 2023 by qyp422, All Rights Reserved. 
'''
import base,os,sys,dumpreader,mathfuction
def main():
    if len(sys.argv)< 5:
        sys.exit('Syntax: $> python get_contactmap.py input_filename top_filename output_filename framenum')
    filename = os.path.basename(sys.argv[1])
    topfile = os.path.basename(sys.argv[2])
    output_filename = sys.argv[3]
    framenum = int(sys.argv[4])

    r = dumpreader.Lammps_dumpreader(filename,topfile)
    s = r.get_frame_system(framenum)
    data = s.get_contactmap(s._strands[0],s._strands[1])
    mathfuction.get_supercoiling_shape(data)
    base.Plot_system.plot_contactmap(out_filename=output_filename,data=data)

if __name__ == "__main__":
        main()
