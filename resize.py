from skimage import data, io, filters
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from skimage.transform import rescale, resize, downscale_local_mean

N=190 #number of images

for i in range (1,N+1):
    if i < 10 :
        path = '00'+str(i)+'.bmp'
    elif i < 100:
        path = '0'+str(i)+'.bmp'
    elif i >= 100:
        path = str(i)+'.bmp'
    print(path)
    
    image= io.imread('C:/Users/Malikoto/unet-master/data/Dataset/train/origin image/'+path)
    image = resize(image, (100,100))

    path=str(i)+'.png'
    
    io.imsave('C:/Users/Malikoto/unet-master/data/Dataset/train/resize image/'+path,image)
