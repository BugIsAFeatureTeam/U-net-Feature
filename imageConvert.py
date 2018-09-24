from skimage import io

#Convert .bmp dataset to .tif

#Number of images
n=210 
for i in range (1,n+1):
#    if i < 10 :
#        path = '00'+str(i)+'.png'
#    elif i < 100:
#        path = '0'+str(i)+'.png'
#    elif i >= 100:
#        path = str(i)+'.png'
#        
    path = str(i)+'.png'
    print(path)
    
    img= io.imread('C:/Users/Malikoto/unet-master/data/Dataset/train/resize image/'+path)
    print(img.shape)
    
    path=str(i)+'.tif'
    io.imsave('C:/Users/Malikoto/unet-master/data/Dataset/train/tif image/'+path,img)
