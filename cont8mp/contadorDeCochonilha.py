# -*- coding: utf-8 -*-

import cv2
import mahotas
import numpy as np

#img = cv2.imread('img_palma.png')
img = cv2.imread("8mp.JPG")


height, width = img.shape[:2]
#width = img.shape[0]
#height = img.shape[1]
 
#height = np.size(img, 0)
#width = np.size(img, 1)

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64) 
fgdModel = np.zeros((1,65),np.float64)

rect = (10,10,width-30,height-30)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img2 = img*mask[:,:,np.newaxis] 

background = cv2.subtract(img, img2)

#Alterar os pixels que não são pretos para brancos
background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]

frontendColorido = background + img2

#cv2.imshow("Frontend", frontendColorido)

frontend = cv2.cvtColor(frontendColorido, cv2.COLOR_BGR2GRAY)

suave = cv2.GaussianBlur(frontend, (7, 7), 0) # aplica blur

(T, bin) = cv2.threshold(suave, 167, 255,cv2.THRESH_BINARY_INV)

bordas = cv2.Canny(bin, 70, 150)

#cv2.imshow("Identificando as bordas", bordas)

(lx, objetos, lx) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imgC2 = img.copy()

cv2.imshow("Imagem Original", img)
cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)

fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgC2, str(len(objetos))+" Cochonilhas encontradas!", (10,20), fonte, 0.5, (255,0,0), 0, cv2.LINE_AA)

cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)


'''
imagem = cv2.imread("palma/original.png", 0)
#imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(imagem, (7, 7), 0) # aplica blur

T = mahotas.thresholding.otsu(suave)

temp = imagem.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)

T = mahotas.thresholding.rc(suave)

temp2 = imagem.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)

resultado = np.vstack([
    np.hstack([imagem, suave]),
    np.hstack([temp, temp2])
])


cv2.imshow("Suave", resultado)
cv2.imwrite('04.png', resultado)
cv2.waitKey(0)
'''

'''

bin1 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

resultado = np.vstack([
    np.hstack([imagem, suave]),
    np.hstack([bin1, bin2])
]) 



T = mahotas.thresholding.otsu(suave)

temp = imagem.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)

T = mahotas.thresholding.rc(suave)

temp2 = imagem.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)

resultado = np.vstack([
    np.hstack([imagem, suave]),
    np.hstack([temp, temp2])
])-





--------------- INPORTANTE --------------
import cv2
import mahotas
import numpy as np


imgColorida = cv2.imread("subtract.png")

img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)

suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur

(T, bin) = cv2.threshold(suave, 167, 255,cv2.THRESH_BINARY_INV)

bordas = cv2.Canny(bin, 70, 150)

(lx, objetos, lx) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imgC2 = imgColorida.copy()

cv2.imshow("Imagem Original", imgColorida)
cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)

fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgC2, str(len(objetos))+" Cochonilhas encontradas!", (10,20), fonte, 0.5, (255,0,0), 0, cv2.LINE_AA)

cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)

-----------------------------------------



'''