#!/home/daniel/anaconda2/bin/ipython

# -*- coding: utf-8 -*-
"""

Core of the LAGAR package. This script reads and connects the differents
inputs and outputs of the different modules. It works as a template
and it can be modified.

@author: Angel Daniel Garaboa Paz

@mail: angeldaniel.garaboa@usc.es

"""

from LAGAR.LAGARCore import LAGARCore, LAGAR_Parallel

import time
import glob
import sys
import json
import os


# This function is just for pretty print the dictionaries.

def pprint(myDict):
    print('\n'.join("{:<20} {:<20}".format(k, v) for k, v in myDict.items()))

# Pretty print the presentation. If you have the toilet package installed under
# Linux distribution the presentation will be better. Thats'all


def main():

	os.system('toilet -f smblock --filter border:metal "Welcome to LAGAR!"')
	os.system('toilet -f future "GFNL, USC, Santiago, Spain"')
	os.system('echo "author: Angel Daniel Garaboa Paz"')
	os.system('echo "author: Vicente Pérez Muñuzuri"')

	print('')
	print('*-----------*')
	print('|SETUP FILE:|', sys.argv[1], sys.argv[2])
	print('*-----------*')
	print('')


	if int(sys.argv[2]) > 1:
		LAGAR_father = LAGAR_Parallel(sys.argv[1])
		LAGAR_father.split_time_chunks(int(sys.argv[2]))
		LAGAR_father.set_father_simulation()

		for time_section in LAGAR_father.time_intervals:
		   
		    LAGAR_father.update_father(time_section)

	else:
		LAGAR_father = LAGARCore(sys.argv[1])
		LAGAR_father.read_simulation()
		LAGAR_father.set_simulation()
		LAGAR_father.run_simulation()


if __name__ == "__main__":
    # execute only if run as a script
    main()



#LAGAR_sons = [LAGAR_Parallel.update_son(LAGAR_son,time_section) for LAGAR_son in LAGAR_sons]
#pool.map(LAGAR_Parallel.run_son, LAGAR_sons)



#blocks_vtu = []
#for block in range(0,4):
#    blocks_vtu.append(sorted(glob('./block_'+str(block).zfill(2)+'*')))
#
#blocks_vtu = list(zip(*blocks_vtu))
#t = 0
#for block in blocks_vtu: 
#    file = open('t_'+str(t)+'.pvtu',"w")
#    file.write("<?xml version=\"1.0\"?>\n")
#    file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
#    file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
#    file.write("<PPoints>\n")
#    file.write("<PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/>\n")
#    file.write("</PPoints>\n")
#    
#    for vtu in block:
#        file.write("<Piece Source=\"" + vtu +"\"/>\n")
#    
#    file.write("</PUnstructuredGrid>\n")
#    file.write("</VTKFile>\n")
#    t = t+1

    
