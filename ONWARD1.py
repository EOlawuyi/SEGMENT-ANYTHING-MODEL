from PIL import Image as im
import sys
import numpy as np
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import skimage as ski
import skimage.feature
import pandas as pd
from skimage import feature, measure
from skimage.measure import label, regionprops, regionprops_table 
#This is the code that belongs to Olorogun Engineer Enoch O. Ejofodomi in his Collaboration with Shell.
#This code also belongs to Engineer Francis Olawuyi.


#LOAD IMAGE AND DISPLAY
np.set_printoptions(threshold=sys.maxsize)
data = np.load('test_vol_7.npy',allow_pickle=True)
datab = np.load('test_vol_7.npy',allow_pickle=True)
#testfile = np.load('sub_vol_3.npy',allow_pickle=True)
#print(testfile)
img = im.fromarray(data, 'RGB')
img.save('my.png')
img.show()

print(data.shape)
print(data.size)


#imgb = im.fromarray(data, 'RGB')
#imgb.save('my2.png')
#imgb.show()
data2b = (datab[:,:,2])
#data2 = (data[:][:][2]) - 300 by 300 and SELECTED
print(data2b.shape)
#print(data2b)
#size of array
#data.ndim

SAM1 = input("Enter the GraysScale value of the Pixel you would like to use for SAM: ")
print("The SAM GRAYSCALE PIXEL is: ")
print((SAM1))
SAM = int(SAM1)
print((SAM))
#PRINT DATA SHAPE AND IMAGE
print(data.shape)
[a,b,c] = data.shape
print(a)
print(b)
print(c)


#SELECT DATA SLICE
data2 = (data[:,:,2])
#data2 = (data[:][:][2]) - 300 by 300 and SELECTED
print(data2.shape)
#print(data2)

data3 = (data[:,2,:])
#data2 = (data[:][:][2]) -  300 BY 100 NOT SELECTED
print(data3.shape)
#print(data3)

data4 = (data[2,:,:])
#data2 = (data[:][:][2]) - 300 BY 100 NOT SELECTED
print(data4.shape)
#print(data4)

data5= np.zeros((300,300))
# Convert Image to Grayscale - FIRST TEST
img5 = im.fromarray(np.uint8(data3 * 255), 'L')
img5.show()

# convert data file to an image file. But data rannge of -1 to 1 is not
#acceptable for a grayscale image conversion.
#img6 = np.interp(data2, (data2.min(), data2.max()), (-1,+1))
#img7 = im.fromarray(np.uint8(img6 * 255), 'L')
#img7.show()
print('labels')
#Proper conversion of data file to an image file in grayscale (FINAL)
img8 = np.interp(data3, (data3.min(), data3.max()), (0,+1))
img9 = im.fromarray(np.uint8(img8 * 255), 'L')
img9.show()

img8b = np.interp(data2b, (data2b.min(), data2b.max()), (0,+1))
img9b = im.fromarray(np.uint8(img8b * 255), 'L')
img9b.show()

[d,e] = img9.size
print(d)
print(e)
img10 = im.fromarray(np.uint8(img8 * 255), 'L')
img11 = im.fromarray(np.uint8(img8 * 255), 'L')
img12 = im.fromarray(np.uint8(img8 * 255), 'L')
img13 = im.fromarray(np.uint8(img8 * 255), 'L')
img14 = im.fromarray(np.uint8(img8 * 255), 'L')
img15a = im.fromarray(np.uint8(img8 * 255), 'L')
img15b = im.fromarray(np.uint8(img8 * 255), 'L')
img15c = im.fromarray(np.uint8(img8 * 255), 'L')
img15d = im.fromarray(np.uint8(img8 * 255), 'L')
img16 = im.fromarray(np.uint8(img8 * 255), 'L')


#Grayscale Thresholding 1
for i in range(1,d-1):
    for j in range(1,e-1):
        if( img10.getpixel((i,j)) > 180):
           img15a.putpixel((i,j), 255)
        else:
            img15a.putpixel((i,j), 0)
print('Color Homogeneity 1')    
img15a.show(title='HOMOGENEITY')


#perform Region Props on Thresholded Image
img15aa = np.array(img15a)
#Select Pixels Greater than 100 with a mask
mask = img15aa > 100
labels = measure.label(mask)

#Segment out Regions
regions = measure.regionprops(labels, img15aa)
numlabels = len(regions)
regions = regionprops_table(labels, properties=('area', 'coords'))
print(regions)
pd.DataFrame(regions)
y = pd.DataFrame(regions)
#Get Shape and Size of Regions
print(y.shape)


[a1,b1] = y.shape

#Print out Region Details
ttt = np.zeros((a1,b1))

#Select Only Regions Greater than 500 Pixels and Get their Index Number
count =0
gray1 = np.zeros((5,2))

for i in range(a1):
    if(y.values[i,0] > 500):
       print('found one!')
       if(count < 5):
           gray1[count,0] = i
           gray1[count,1] = y.values[i,0]
           count = count+1
       
print(gray1)

y.values[int(gray1[0,0]),1][1,0]    
[shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
[shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
[shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
[shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
[shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

print(shape1a)
print(shape1b)


threshold0 = np.zeros((d,e))
if(gray1[0,0] != 0):
    for i in range(0, shape1a-2):
        xco = y.values[int(gray1[0,0]),1][i,0]
        yco = y.values[int(gray1[0,0]),1][i,1]
        threshold0[xco,yco] = 1
if(gray1[1,0] != 0):
    for i in range(0, shape2a-2):
        xco = y.values[int(gray1[1,0]),1][i,0]
        yco = y.values[int(gray1[1,0]),1][i,1]
        threshold0[xco,yco] = 1
if(gray1[2,0] != 0):
    for i in range(0, shape3a-2):
       xco = y.values[int(gray1[2,0]),1][i,0]
       yco = y.values[int(gray1[2,0]),1][i,1]
       threshold0[xco,yco] = 1
if(gray1[3,0] != 0):
    for i in range(0, shape4a-2):
        xco = y.values[int(gray1[3,0]),1][i,0]
        yco = y.values[int(gray1[3,0]),1][i,1]
        threshold0[xco,yco] = 1
if(gray1[4,0] != 0):
    for i in range(0, shape5a-2):
        xco = y.values[int(gray1[4,0]),1][i,0]
        yco = y.values[int(gray1[4,0]),1][i,1]
        threshold0[xco,yco] = 1

print(shape1a)
print(shape1b)
               
print(threshold0)



#Grayscale Thresholding 2
for i in range(1,d-1):
    for j in range(1,e-1):
        if( ( img10.getpixel((i,j)) > 100) &  (img10.getpixel((i,j)) < 120)):
           img15b.putpixel((i,j), 255)
           data5[i,j] = 255
        else:
            img15b.putpixel((i,j), 0)
            data5[i,j] = 0

labeled_image1 = ski.measure.label(data5, connectivity=2, return_num=True)
labeled_image2, count = ski.measure.label(data5, connectivity=2, return_num=True)

print('RegionProps 1')

#perform Region Props on Thresholded Image
img15bb = np.array(img15b)
#Select Pixels Greater than 100 with a mask
mask = img15bb > 100
labels = measure.label(mask)

#Segment out Regions
regions = measure.regionprops(labels, img15bb)
numlabels = len(regions)
regions = regionprops_table(labels, properties=('area', 'coords'))
print(regions)
pd.DataFrame(regions)
y = pd.DataFrame(regions)
#Get Shape and Size of Regions
print(y.shape)
print(y.size)
[a1,b1] = y.shape

#Print out Region Details
ttt = np.zeros((a1,b1))
#y.index
#RangeIndex(start=0, stop=1692, step=1)
#y.columns
#Index(['area', 'perimeter'], dtype='object')
#y.columns[1]
#'perimeter'
#y.columns[0]
#'area'
#y.max()
#area         862.000000
#perimeter    645.558441
#dtype: float64
#y.min()
#area         1.0
#perimeter    0.0
#dtype: float64
#y.idxmax()
#area         1292
#perimeter    1292
#dtype: int64
#y.idxmin()
#area         0
#perimeter    0
#dtype: int64
#y.values[1,1]
#y.values[b,c], where b is row (from 0 to end)and c is column (from 0 to end)
#y.values[1,1][1,0]
#y.values[b,c][d,e], where b is row (from 0 to end)and c is column (from 0 to end)
#for column 1 (2nd cloumn) you have a list of  coordinates (x & y) for all the
#points in a region. Then subsequently, [d,e], represents d - row number for
#the coordicnates of the pixels in the list for the region
#e - the colum for the coordinates of the pixels in the list for the region
#again, the column is either 0 (for x coordinates) or 1 (for y coordinates)

#Select Only Regions Greater than 500 Pixels and Get their Index Number
count =0
gray1 = np.zeros((5,2))

for i in range(a1):
    if(y.values[i,0] > 500):
       print('found one!')
       if(count < 5):
           gray1[count,0] = i
           gray1[count,1] = y.values[i,0]
           count = count+1
       
print(gray1)

y.values[int(gray1[0,0]),1][1,0]    
[shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
[shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
[shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
[shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
[shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape



print(shape1a)
print(shape1b)

threshold1 = np.zeros((d,e))
if(gray1[0,0] != 0):
    for i in range(0, shape1a-2):
        xco = y.values[int(gray1[0,0]),1][i,0]
        yco = y.values[int(gray1[0,0]),1][i,1]
        threshold1[xco,yco] = 2

if(gray1[1,0] != 0):
    for i in range(1, shape2a-2):
        xco = y.values[int(gray1[1,0]),1][i,0]
        yco = y.values[int(gray1[1,0]),1][i,1]
        threshold1[xco,yco] = 2

if(gray1[2,0] != 0):
    for i in range(1, shape3a-2):
       xco = y.values[int(gray1[2,0]),1][i,0]
       yco = y.values[int(gray1[2,0]),1][i,1]
       threshold1[xco,yco] = 2
if(gray1[3,0] != 0):
    for i in range(1, shape4a-2):
        xco = y.values[int(gray1[3,0]),1][i,0]
        yco = y.values[int(gray1[3,0]),1][i,1]
        threshold1[xco,yco] = 2
if(gray1[4,0] != 0):
    for i in range(1, shape5a-2):
        xco = y.values[int(gray1[4,0]),1][i,0]
        yco = y.values[int(gray1[4,0]),1][i,1]
        threshold1[xco,yco] = 2

    
print(shape2a)
print(shape2b)               
print(threshold1)

print('Color Homogeneity 2')    

#img11.save("img12.png")
img15b.show(title='HOMOGENEITY')


#Grayscale Thresholding 3
for i in range(1,d-1):
    for j in range(1,e-1):
        if( ( img10.getpixel((i,j)) > 50) &  (img10.getpixel((i,j)) < 80)):
           img15c.putpixel((i,j), 255)
        else:
            img15c.putpixel((i,j), 0)
print('Color Homogeneity 3')    
#img11.save("img12.png")
img15c.show(title='HOMOGENEITY')



#perform Region Props on Thresholded Image
img15cc = np.array(img15c)
#Select Pixels Greater than 100 with a mask
mask = img15cc > 100
labels = measure.label(mask)

#Segment out Regions
regions = measure.regionprops(labels, img15cc)
numlabels = len(regions)
regions = regionprops_table(labels, properties=('area', 'coords'))
print(regions)
pd.DataFrame(regions)
y = pd.DataFrame(regions)
#Get Shape and Size of Regions
print(y.shape)
print(y.size)
[a1,b1] = y.shape

#Print out Region Details
ttt = np.zeros((a1,b1))

#Select Only Regions Greater than 500 Pixels and Get their Index Number
count =0
gray1 = np.zeros((5,2))

for i in range(a1):
    if(y.values[i,0] > 500):
       print('found one!')
       if(count < 5):
           gray1[count,0] = i
           gray1[count,1] = y.values[i,0]
           count = count+1
       
print(gray1)

y.values[int(gray1[0,0]),1][1,0]    
[shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
[shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
[shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
[shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
[shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

print(shape1a)
print(shape1b)


threshold2 = np.zeros((d,e))
if(gray1[0,0] != 0):
    for i in range(0, shape1a-1):
        xco = y.values[int(gray1[0,0]),1][i,0]
        yco = y.values[int(gray1[0,0]),1][i,1]
        threshold2[xco,yco] = 3
if(gray1[1,0] != 0):
    for i in range(0, shape2a-1):
        xco = y.values[int(gray1[1,0]),1][i,0]
        yco = y.values[int(gray1[1,0]),1][i,1]
        threshold2[xco,yco] = 3
if(gray1[2,0] != 0):
    for i in range(0, shape3a-1):
       xco = y.values[int(gray1[2,0]),1][i,0]
       yco = y.values[int(gray1[2,0]),1][i,1]
       threshold2[xco,yco] = 3
if(gray1[3,0] != 0):
    for i in range(0, shape4a-1):
        xco = y.values[int(gray1[3,0]),1][i,0]
        yco = y.values[int(gray1[3,0]),1][i,1]
        threshold2[xco,yco] = 3
if(gray1[4,0] != 0):
    for i in range(0, shape5a-1):
        xco = y.values[int(gray1[4,0]),1][i,0]
        yco = y.values[int(gray1[4,0]),1][i,1]
        threshold2[xco,yco] = 3

    
               
print(threshold2)

    


#Color Thresholding 4
# Detect Homogeneous Regions in the image by averaginyscal values in a 3 by 3
# matrix and comparing that to the pixel in question (middle).
# if the difference is <10, then the pixel is ima homogeneous pixel
# and it is then assigned a white color (255).
# Otherwise, it is non-homegeous and assigned a black color (0). 

for i in range(1,d-1):
    for j in range(1,e-1):
        average = img11.getpixel((i,j)) + img11.getpixel((i,j+1)) + img11.getpixel((i,j-1))+ img11.getpixel((i-1,j)) + img11.getpixel((i-1,j-1)) + img11.getpixel((i-1,j+1)) + img11.getpixel((i+1,j)) + img11.getpixel((i+1,j-1)) + img11.getpixel((i,j+1))
        average2 = average/9
        if(abs( img11.getpixel((i,j)) - average2) < 10):
           img14.putpixel((i,j), 0)
        else:
            img14.putpixel((i,j), 255)
    
#img11.save("img12.png")
img14.show(title='HOMOGENEITY')


#perform Region Props on Thresholded Image
img14aa = np.array(img14)
#Select Pixels Greater than 100 with a mask
mask = img14aa > 100
labels = measure.label(mask)

#Segment out Regions
regions = measure.regionprops(labels, img14aa)
numlabels = len(regions)
regions = regionprops_table(labels, properties=('area', 'coords'))
print(regions)
pd.DataFrame(regions)
y = pd.DataFrame(regions)
#Get Shape and Size of Regions
print(y.shape)
print(y.size)
[a1,b1] = y.shape

#Print out Region Details
ttt = np.zeros((a1,b1))

#Select Only Regions Greater than 500 Pixels and Get their Index Number
count =0
gray1 = np.zeros((5,2))

for i in range(a1):
    if(y.values[i,0] > 500):
       print('found one!')
       if(count < 5):
           gray1[count,0] = i
           gray1[count,1] = y.values[i,0]
           count = count+1
       
print(gray1)

#y.values[int(gray1[0,0]),1][1,0]    
[shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
[shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
[shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
[shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
[shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

print(shape1a)
print(shape1b)


threshold3 = np.zeros((d,e))
if(gray1[0,0] != 0):
    for i in range(0, shape1a-2):
        xco = y.values[int(gray1[0,0]),1][i,0]
        yco = y.values[int(gray1[0,0]),1][i,1]
        threshold3[xco,yco] = 4
print(shape1a)
print(shape1b)
if(gray1[1,0] != 0):
    for i in range(0, shape2a-2):
        xco = y.values[int(gray1[1,0]),1][i,0]
        yco = y.values[int(gray1[1,0]),1][i,1]
        threshold3[xco,yco] = 4
print(shape1a)
print(shape1b)
if(gray1[2,0] != 0):
    for i in range(0, shape3a-2):
       xco = y.values[int(gray1[2,0]),1][i,0]
       yco = y.values[int(gray1[2,0]),1][i,1]
       threshold3[xco,yco] = 4
print(shape1a)
print(shape1b)
if(gray1[3,0] != 0):
    for i in range(0, shape4a-2):
        xco = y.values[int(gray1[3,0]),1][i,0]
        yco = y.values[int(gray1[3,0]),1][i,1]
        threshold3[xco,yco] = 4
print(shape1a)
print(shape1b)
if(gray1[4,0] != 0):
    for i in range(0, shape5a-2):
        xco = y.values[int(gray1[4,0]),1][i,0]
        yco = y.values[int(gray1[4,0]),1][i,1]
        threshold3[xco,yco] = 4

         
print(threshold3)


#SAM Thresholding 
# Detect SAM Pixels using input provided by the user

for i in range(1,d-1):
    for j in range(1,e-1):
        average = img11.getpixel((i,j)) + img11.getpixel((i,j+1)) + img11.getpixel((i,j-1))+ img11.getpixel((i-1,j)) + img11.getpixel((i-1,j-1)) + img11.getpixel((i-1,j+1)) + img11.getpixel((i+1,j)) + img11.getpixel((i+1,j-1)) + img11.getpixel((i,j+1))
        average2 = average/9
        if(abs( img11.getpixel((i,j)) - SAM) < 10):
           img16.putpixel((i,j), 0)
        else:
            img16.putpixel((i,j), 255)
    
#img11.save("img12.png")
img16.show(title='SAM PIXELS')


#perform Region Props on Thresholded Image
img16aa = np.array(img16)
#Select Pixels Greater than 100 with a mask
mask = img16aa > SAM-10
labels = measure.label(mask)

#Segment out Regions
regions = measure.regionprops(labels, img16aa)
numlabels = len(regions)
regions = regionprops_table(labels, properties=('area', 'coords'))
print(regions)
pd.DataFrame(regions)
y = pd.DataFrame(regions)
#Get Shape and Size of Regions
print(y.shape)
print(y.size)
[a1,b1] = y.shape

#Print out Region Details
ttt = np.zeros((a1,b1))

#Select Only Regions Greater than 500 Pixels and Get their Index Number
count =0
gray1 = np.zeros((5,2))

for i in range(a1):
    if(y.values[i,0] > 500):
       print('found one!')
       if(count < 5):
           gray1[count,0] = i
           gray1[count,1] = y.values[i,0]
           count = count+1
       
print(gray1)

y.values[int(gray1[0,0]),1][1,0]    
[shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
[shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
[shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
[shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
[shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

print(shape1a)
print(shape1b)


threshold4 = np.zeros((d,e))
if(gray1[0,0] != 0):
    for i in range(0, shape1a):
        xco = y.values[int(gray1[0,0]),1][i,0]
        yco = y.values[int(gray1[0,0]),1][i,1]
        threshold4[xco,yco] = 5
if(gray1[1,0] != 0):
    for i in range(0, shape2a):
        xco = y.values[int(gray1[1,0]),1][i,0]
        yco = y.values[int(gray1[1,0]),1][i,1]
        threshold4[xco,yco] = 5
if(gray1[2,0] != 0):
    for i in range(0, shape3a):
       xco = y.values[int(gray1[2,0]),1][i,0]
       yco = y.values[int(gray1[2,0]),1][i,1]
       threshold4[xco,yco] = 5
if(gray1[3,0] != 0):
    for i in range(0, shape4a):
        xco = y.values[int(gray1[3,0]),1][i,0]
        yco = y.values[int(gray1[3,0]),1][i,1]
        threshold4[xco,yco] = 5
if(gray1[4,0] != 0):
    for i in range(0, shape5a):
        xco = y.values[int(gray1[4,0]),1][i,0]
        yco = y.values[int(gray1[4,0]),1][i,1]
        threshold4[xco,yco] = 5

         



    
#final image
for i in range(0,d-1):
    for j in range(0,e-1):
        img12.putpixel((i,j), 0)
        if(threshold0[j,i] == 1):
            img12.putpixel((i,j), 255)
        if(threshold1[j,i] == 2):
            img12.putpixel((i,j), 255)
        if(threshold2[j,i] == 3):
            img12.putpixel((i,j), 255)
        if(threshold3[j,i] == 4):
            img12.putpixel((i,j), 255)
        if(threshold4[j,i] == 5):
            img12.putpixel((i,j), 255)
      
img12.show(title='LAYERS')
img12.show(title="LAYERS")
print(img12)
sub_vol_7 = np.array(img12)
imageppp = np.asanyarray(img12)
print(imageppp)
np.save('sub_vol_7.npy',sub_vol_7)
#np.savetxt('sub_vol_69774952.npy',sub_vol_69774952)

#69774755

#img16 = img12.transpose()

#img16.show(title='LAYERS 2')
#imageppp = np.asanyarray(img13)
#edge=feature.canny(imageppp, sigma=1)

#edges = skimage.feature.canny(image=imageppp, sigma=0.5, low_threshold=0.1, high_threshold=0.3)
#PRODUCES 1 LAYER

#edges2 = skimage.feature.canny(image=imageppp, sigma=0.1, low_threshold=0.05, high_threshold=0.7)

#skimage.io.imshow(edges)

#edges.view()
#[f,g] = edges.size

#for i in range(0,d-1):
#    for j in range(0,e-1):
      
#        if(( edges2[i,j]) == False):
#           img14.putpixel((j,i), 0)
#        else:
#            img14.putpixel((j,i), 255)

#img14.show(title='EDGE')

 




#label_img = label(img8)
#regions = regionprops(label_img)
#plt.imshow(label_img)
#<matplotlib.image.AxesImage object at 0x00000228E62884D0>
#plt.show()
#plt.colorbar()

#labeled_image = ski.measure.label(img15a, connectivity=2)
#covert numpy array to gryscale
#grayscaleimage = rgb2gray(array)


#covert grayscale to numpy array
#imagArray = np.array(imgData)

#skimage.measure.regionprops(label)imahge, intensity_image=None, cache= True, *,extra_properties=None, spacing=Nooooo, offset=None)
#Detect Lines in the image by chacking on the grayscale value of the
#neighboring pixel

             
#edges = skimage.feature.canny(image=image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

# denoise the image with a Gaussian filter
    #blurred_image = ski.filters.gaussian(gray_image, sigma=sigma)
    # mask the image according to threshold
   #  binary_mask = blurred_image < t
    # perform connected component analysis
#labeled_image, count = ski.measure.label(img13, connectivity=2, return_num=True)

#blurred_image = ski.filters.gaussian(img13, sigma=1.0)
    # mask the image according to threshoprinld
#binary_mask = blurred_image < 0.5
    # perform connected component analysis
#labeled_image, count = ski.measure.label(binary_mask, connectivity=2, return_num=True)



#Loop through Image File to search for Horizaontal Layers
  #  aaa=1;
   # for row in range(foundpatientrow2 + n + 1, foundpatientrow2 + n + 2):
    #    dataframe1.cell(row,1).value=data1

    #for row in range(0, dataframe1.max_row):
     #   if(yfoundpatient == 0):
      #      for col in dataframe1.iter_cols(1, 1):
       #         efile = (col[row].value)



 #for row in range(foundpatientrow2, dataframe1.max_row):
  #                       for col in dataframe1.iter_cols(1, dataframe1.max_column):
   #                          if(row <= (foundpatientrow2 + n - 1)):
    #                             print(col[row].value)
     #                            canvas5.create_text(numarray[p], numarray2, text=col[row].value, width=600, font=('Arial bold', 20), fill='#000000')
      #                           p=p+1



#img7 = np.interp(data2, (data2.min(), data2.max()), (-1,+1))
#img7.show()
# scale vector to [0,1] range, multipluy by 255 and convert to uint8
#new_arr = ((data4 - data4.min()) * (1/data4.max() - data4.min()) * 255).astype('uint8')
#print(new_arr.shape)000
#imga = im.fromarray(data2, 'L')
#print(imga.size)
#print(imga.shape)
#imga.show

#data3 = (data[:][2][:]).squeeze()
#print(data3.shape)
#imgb = im.fromarray(data3, 'L')
#imgb.show
#print(imgb.size)
#print(imgb.shape)

#data4 = (data[2][:][:]).squeeze()
#print(data4.shape)
#imgc = im.fromarray(data4, 'L')
#imgc.show
#print(imgc.size)
#print(imgc.shape)
#img2 = im.fromarray(np.uint8(tmp))
#img2.save('my.png')
#img2.show()
#data3 = (data[:][2][:]).squeeze()
#print(data3.shape)
#img3 = im.fromarray(data3)
#img3.save('my.png')
#img3.show()

 
# scale vector to [0,1] range, multipluy by 255 and convert to uint8
#new_arr = ((arr - arr.min()) * (1/arr.max() - arr.min()) * 255)).astype('uint8')



#import numpy as np
#from PIL import Image

# create random image
# mat = np.random.random((100,100))
#img = Image.fromarray(mat, 'L')
#img.show


#import numpy as np
#from PIL import Image

# gradient between 0 and 1 for 256 *256
# array = np.linespace(0,1,256*256)


#reshape to 2d
#mat = np.reshape(array, (256,256))

#create PIL Image
#img = Image.fromarray(np.uint8(mat * 255), 'L')
#img.show()








   # 
   # print(label_i)
   # pro
       
#for region in regions:
 #   region_id = region_label
  #  area = region.area
   # x0 = props.area
   # if props.area >500:
    #    print('foound one!')

#for index in range(1, labels.max()):
#    if(labels.[index].area_i = props[index].label
#    print(label_i)
#    pro
    
 #   if(props[index].area > 500):
  #      print('found one!')
    
    
#props = measure.regionprops(img15bb, img15bb)
#props = regionprops_table(img15bb, properties=('area'))
#print(props)                        

#)
#print(labels)



#label_img = label(img15bb)

  
#regions = regionprops(label_img)
#plt.imshow(label_img)
#plt.show()
#label_img = label(img15bb)
#regions = regionprops(label_img)
#plt.imshow(label_img)
#plt.show()
#props = measure.regionprops(labels, img)
#properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']



#Print out Region Details
ttt = np.zeros((a1,b1))
#y.index
#RangeIndex(start=0, stop=1692, step=1)
#y.columns
#Index(['area', 'perimeter'], dtype='object')
#y.columns[1]
#'perimeter'
#y.columns[0]
#'area'
#y.max()
#area         862.000000
#perimeter    645.558441
#dtype: float64
#y.min()
#area         1.0
#perimeter    0.0
#dtype: float64
#y.idxmax()
#area         1292
#perimeter    1292
#dtype: int64
#y.idxmin()
#area         0
#perimeter    0
#dtype: int64
#y.values[1,1]
#y.values[b,c], where b is row (from 0 to end)and c is column (from 0 to end)
#y.values[1,1][1,0]
#y.values[b,c][d,e], where b is row (from 0 to end)and c is column (from 0 to end)
#for column 1 (2nd cloumn) you have a list of  coordinates (x & y) for all the
#points in a region. Then subsequently, [d,e], represents d - row number for
#the coordicnates of the pixels in the list for the region
#e - the colum for the coordinates of the pixels in the list for the region
#again, the column is either 0 (for x coordinates) or 1 (for y coordinates)



#for i in range(0,d):
#    for j in range(1,e):
#        if(( img10.getpixel((i,j)) - img10.getpixel((i-1,j))) > 20):
#           img11.putpixel((i,j), 255)
#img11.show()


#Detect Lines in the image by chacking on the grayscale value of the
#neighboring pixel
#Pixels that are not identified as lines are assigned a black color (0)
# so that the detected lines in the image are visible to the human eye.

#for i in range(0,d):
#    for j in range(1,e):
#        if(( img10.getpixel((i,j)) - img10.getpixel((i-1,j))) > 20):
#           img12.putpixel((i,j), 255)
#        else:
#            img12.putpixel((i,j), 0)
#img12.show(title='LLNE DETECTOR')   
