from skimage import data, io, filters
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from skimage.transform import rescale, resize, downscale_local_mean

def toBinary(path):
    image= io.imread('C:/Users/Malikoto/unet-master/data/Dataset/train/label/'+path,as_grey=True)
    print('b4',len(image[0]), len(image))
    
    for i in range(0,len(image)):
        for j in range(0,len(image[0])):
            if image[i][j] == 0:
                image[i][j] = 255
            elif image[i][j] == 128:
                image[i][j] = 0
            elif image[i][j] == 255:
                image[i][j] = 255
                
    image = resize(image, (100,100))
    print('after',len(image[0]), len(image))
    
    return image

N=210 #number of images

for i in range (1,N+1):
    if i < 10 :
        path = '00'+str(i)+'.png'
    elif i < 100:
        path = '0'+str(i)+'.png'
    elif i >= 100:
        path = str(i)+'.png'
    print(path)
    
    binary = toBinary(path)
    
#    plt.gray()
#    plt.imshow(binary)
#    plt.show()
#    print(binary.shape)
    path=str(i)+'.png'
    
    io.imsave('C:/Users/Malikoto/unet-master/data/Dataset/train/tif label/'+path,binary)
