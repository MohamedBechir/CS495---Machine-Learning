import numpy as np
import matplotlib.pyplot as plt


#   DATA
# Desired Output
yd=np.array([5,1,1,4])
x=np.array([[3,-1,2,7],[-7,1,3,3],[13,0,4,9]]).reshape(3,4)
w=np.array([0,0,0]).reshape(3,1)
lamd=10
delt=.01
epoch=300
T=np.zeros(epoch)
Yt1=np.zeros(epoch)
Yt2=np.zeros(epoch)
Yt3=np.zeros(epoch)
Yt4=np.zeros(epoch)
for i in range(epoch):
#feeforward
  for j in range(4):
      y=np.dot(np.transpose(w),x[:,j])
      print(y)
      print('----------')
# weighet update
      dw=(delt*lamd*x[:,j]/(np.dot(np.transpose(x[:,j]),x[:,j]))*(y-yd[j])).reshape(3,1)
      w=w-dw
      if j == 0:
        Yt1[i]=y
      elif j  == 1:
        Yt2[i]=y
      elif j  == 2:
        Yt3[i]=y
      elif j  == 3:
        Yt4[i]=y
  T[i]=i

plt.title("Neurone output")
plt.ylabel("Amplitude")
plt.xlabel("epoch")
plt.plot(T,Yt1)
plt.plot(T,Yt2)
plt.plot(T,Yt3)
plt.plot(T,Yt4)
plt.show()