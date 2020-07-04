# Image Classificator

Go AI image classificator from scratch

The objective is to tell apart potato images from carrot images using go and artifical inteligence.
I will be using no libs, only knowledge I can get from the internet for free. 
This is supposed to be an introduction to neural networks so I can complete [this project.](https://github.com/renatobrittoaraujo/rocketlander)
I've been consolidating my linear algebra and calculus skills for a month now, so I can jump to machine learning without lacking math.

## Structure

In dataset, you can find two folder containing potatoes and carrots. That is the dataset.
I extracted it all from google using the code in utils.

In utils, you can find useful tools to build a dataset on your own.

### Building up a dataset

Go to google and type what you want the dataset to be about. 
Go to images, scroll down the page until you think it's enough then save the .html file to util folder.
Now run link_parser.py and input the file name and the folder you want the .jpg pictures to download to.
Now run renamer.py to rename the files in increasing order from [1..n].
