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


np.set_printoptions(threshold=sys.maxsize)
data = np.load('test_vol_50.npy',allow_pickle=True)
datab = np.load('test_vol_50.npy',allow_pickle=True)

img = im.fromarray(data, 'RGB')

data2b = (datab[:,:,2])


#SAM1 = input("Enter the GraysScale value of the Pixel you would like to use for SAM: ")
#print("The SAM GRAYSCALE PIXEL is: ")
#print((SAM1))
#SAM = int(SAM1)
SAM = int(100)
finaldata= np.zeros((300,300,100))

for l in range(0,100):
    #SELECT DATA SLICE
    data2 = (data[:,:,l])

    data5= np.zeros((300,300))
    img5 = im.fromarray(np.uint8(data2 * 255), 'L')

    img8 = np.interp(data2, (data2.min(), data2.max()), (0,+1))
    img9 = im.fromarray(np.uint8(img8 * 255), 'L')

    img8b = np.interp(data2b, (data2b.min(), data2b.max()), (0,+1))
    img9b = im.fromarray(np.uint8(img8b * 255), 'L')

    [d,e] = img9.size
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


    for i in range(1,d-1):
        for j in range(1,e-1):
            if( img10.getpixel((i,j)) > 180):
               img15a.putpixel((i,j), 255)
            else:
                img15a.putpixel((i,j), 0)   
    img15aa = np.array(img15a)
    mask = img15aa > 100
    labels = measure.label(mask)
    regions = measure.regionprops(labels, img15aa)
    numlabels = len(regions)
    regions = regionprops_table(labels, properties=('area', 'coords'))
    pd.DataFrame(regions)
    y = pd.DataFrame(regions)
    [a1,b1] = y.shape

    count =0
    gray1 = np.zeros((5,2))

    for i in range(a1):
        if(y.values[i,0] > 500):
           if(count < 5):
               gray1[count,0] = i
               gray1[count,1] = y.values[i,0]
               count = count+1
        
    [shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
    [shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
    [shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
    [shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
    [shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape


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

    for i in range(1,d-1):
        for j in range(1,e-1):
            if( ( img10.getpixel((i,j)) > 100) &  (img10.getpixel((i,j)) < 120)):
               img15b.putpixel((i,j), 255)
               data5[i,j] = 255
            else:
                img15b.putpixel((i,j), 0)
                data5[i,j] = 0

    img15bb = np.array(img15b)
    mask = img15bb > 100
    labels = measure.label(mask)
    regions = measure.regionprops(labels, img15bb)
    numlabels = len(regions)
    regions = regionprops_table(labels, properties=('area', 'coords'))
    pd.DataFrame(regions)
    y = pd.DataFrame(regions)
    [a1,b1] = y.shape

    count =0
    gray1 = np.zeros((5,2))

    for i in range(a1):
        if(y.values[i,0] > 500):
           if(count < 5):
               gray1[count,0] = i
               gray1[count,1] = y.values[i,0]
               count = count+1
             
    [shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
    [shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
    [shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
    [shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
    [shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape


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


    for i in range(1,d-1):
        for j in range(1,e-1):
            if( ( img10.getpixel((i,j)) > 50) &  (img10.getpixel((i,j)) < 80)):
               img15c.putpixel((i,j), 255)
            else:
                img15c.putpixel((i,j), 0)
    img15cc = np.array(img15c)
    mask = img15cc > 100
    labels = measure.label(mask)
    regions = measure.regionprops(labels, img15cc)
    numlabels = len(regions)
    regions = regionprops_table(labels, properties=('area', 'coords'))
    pd.DataFrame(regions)
    y = pd.DataFrame(regions)
    [a1,b1] = y.shape
    count =0
    gray1 = np.zeros((5,2))

    for i in range(a1):
        if(y.values[i,0] > 500):
           if(count < 5):
               gray1[count,0] = i
               gray1[count,1] = y.values[i,0]
               count = count+1
     
    [shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
    [shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
    [shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
    [shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
    [shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

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

    for i in range(1,d-1):
        for j in range(1,e-1):
            average = img11.getpixel((i,j)) + img11.getpixel((i,j+1)) + img11.getpixel((i,j-1))+ img11.getpixel((i-1,j)) + img11.getpixel((i-1,j-1)) + img11.getpixel((i-1,j+1)) + img11.getpixel((i+1,j)) + img11.getpixel((i+1,j-1)) + img11.getpixel((i,j+1))
            average2 = average/9
            if(abs( img11.getpixel((i,j)) - average2) < 10):
               img14.putpixel((i,j), 0)
            else:
                img14.putpixel((i,j), 255)
    img14aa = np.array(img14)
    mask = img14aa > 100
    labels = measure.label(mask)
    regions = measure.regionprops(labels, img14aa)
    numlabels = len(regions)
    regions = regionprops_table(labels, properties=('area', 'coords'))
    pd.DataFrame(regions)
    y = pd.DataFrame(regions)
    [a1,b1] = y.shape

    count =0
    gray1 = np.zeros((5,2))

    for i in range(a1):
        if(y.values[i,0] > 500):
           if(count < 5):
               gray1[count,0] = i
               gray1[count,1] = y.values[i,0]
               count = count+1

    [shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
    [shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
    [shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
    [shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
    [shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

    threshold3 = np.zeros((d,e))
    if(gray1[0,0] != 0):
        for i in range(0, shape1a-2):
            xco = y.values[int(gray1[0,0]),1][i,0]
            yco = y.values[int(gray1[0,0]),1][i,1]
            threshold3[xco,yco] = 4
    if(gray1[1,0] != 0):
        for i in range(0, shape2a-2):
            xco = y.values[int(gray1[1,0]),1][i,0]
            yco = y.values[int(gray1[1,0]),1][i,1]
            threshold3[xco,yco] = 4
    if(gray1[2,0] != 0):
        for i in range(0, shape3a-2):
           xco = y.values[int(gray1[2,0]),1][i,0]
           yco = y.values[int(gray1[2,0]),1][i,1]
           threshold3[xco,yco] = 4
    if(gray1[3,0] != 0):
        for i in range(0, shape4a-2):
            xco = y.values[int(gray1[3,0]),1][i,0]
            yco = y.values[int(gray1[3,0]),1][i,1]
            threshold3[xco,yco] = 4
    if(gray1[4,0] != 0):
        for i in range(0, shape5a-2):
            xco = y.values[int(gray1[4,0]),1][i,0]
            yco = y.values[int(gray1[4,0]),1][i,1]
            threshold3[xco,yco] = 4

             
    for i in range(1,d-1):
        for j in range(1,e-1):
            average = img11.getpixel((i,j)) + img11.getpixel((i,j+1)) + img11.getpixel((i,j-1))+ img11.getpixel((i-1,j)) + img11.getpixel((i-1,j-1)) + img11.getpixel((i-1,j+1)) + img11.getpixel((i+1,j)) + img11.getpixel((i+1,j-1)) + img11.getpixel((i,j+1))
            average2 = average/9
            if(abs( img11.getpixel((i,j)) - SAM) < 10):
               img16.putpixel((i,j), 0)
            else:
                img16.putpixel((i,j), 255)
        
    img16aa = np.array(img16)
    mask = img16aa > SAM-10
    labels = measure.label(mask)

    regions = measure.regionprops(labels, img16aa)
    numlabels = len(regions)
    regions = regionprops_table(labels, properties=('area', 'coords'))
    pd.DataFrame(regions)
    y = pd.DataFrame(regions)
    [a1,b1] = y.shape

    count =0
    gray1 = np.zeros((5,2))

    for i in range(a1):
        if(y.values[i,0] > 500):
           if(count < 5):
               gray1[count,0] = i
               gray1[count,1] = y.values[i,0]
               count = count+1
           
    [shape1a,shape1b] = y.values[int(gray1[0,0]),1].shape
    [shape2a,shape2b] = y.values[int(gray1[1,0]),1].shape
    [shape3a,shape3b] = y.values[int(gray1[2,0]),1].shape
    [shape4a,shape4b] = y.values[int(gray1[3,0]),1].shape
    [shape5a,shape5b] = y.values[int(gray1[4,0]),1].shape

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


    print(l)
    print('Image Done')
      
#complete image
for i in range(0,d-1):
    for j in range(0,e-1):
        finaldata[i,j,l] =  int(img12.getpixel((i,j)))

#first_test = np.array(img12)

np.save('sub_vol_50.npy',finaldata)
#np.savetxt('sub_vol_50.npy',finaldata)
print('FINAL IMAGE DONE!')




