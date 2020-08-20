# Image Classificator

Go AI image classificator from scratch

The objective is to tell apart potato images from carrot images using go and artifical inteligence.
I will be using no libs, only knowledge I can get in the internet for free. 
This is supposed to be an introduction to neural networks so I can complete [this project.](https://github.com/renatobrittoaraujo/rocketlander)
I've been consolidating my linear algebra and calculus skills for a month now, so I can jump to machine learning and ANNs without lacking math.

This could be done with a few lines of python. But at least now I know what those python lines are doing.

## How to run?

Type in terminal:

```
go build;./img-classificator
```

## Structure

In dataset, you can find two folder containing potatoes and carrots. That is the dataset.
I extracted it all from google using the code in utils.

In utils, you can find useful tools to build a dataset on your own.

### Building up a dataset

Install image handling lib for adapting our dataset:
```
apt install imagemagick
```

Go to google and type what you want the dataset to be about. 
Go to images, scroll down the page until you think it's enough then save the .html file to util folder.
Now run link_parser.py and input the file name and the folder you want the .jpg pictures to download to.
Now run renamer.py to rename the files in increasing order from [1..n].
Now run imageSizeConverter.sh to resize images to a standard format (I use 128x128).
