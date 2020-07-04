import os
import requests # to get image from the web
import shutil # to save it locally

file = input('File name: ')

folder = input('Folder name: ')

if not os.path.exists('./' + folder):
    os.mkdir('./' + folder, 0o755)

file = open(file, 'r')

file_text = file.read()

file_text = file_text.split('https://encrypted-tbn0.gstatic.com/images?q=')

print("NOW DOWLOADING", len(file_text) - 1, "IMAGES")

for i in range(1,len(file_text)): 
    link = 'https://encrypted-tbn0.gstatic.com/images?q=' + file_text[i].split('&amp;')[0]
    print('Dowloading ' + link)
    r = requests.get(link, stream = True)
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(folder + '/' + str(i) + '.jpg','wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',folder + '/' + str(i) + '.jpg')
    else:
        print('Image Couldn\'t be retreived')

file.close()
