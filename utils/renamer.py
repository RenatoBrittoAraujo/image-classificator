import glob
import os
from shutil import copyfile
import shutil
input_folder = input('input folder: ')
output_folder = 'awdaddwadaaaaaaaaaaaaaaaaaaaaaaaaaaaaaadadwadad'
if not os.path.exists('./' + output_folder):
    os.mkdir('./' + output_folder, 0o755)
i = 1
for filename in glob.glob(input_folder + '/*'):
   copyfile(filename, output_folder + '/' + str(i) + '.png')
   i = i + 1
shutil.rmtree(input_folder)
os.mkdir('./' + input_folder, 0o755)
i = 1
for filename in glob.glob(output_folder + '/*'):
   copyfile(filename, input_folder + '/' + str(i) + '.png')
   i = i + 1
shutil.rmtree(output_folder)