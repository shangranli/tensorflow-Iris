import numpy as np
import random



def read_iris(train_size):
    f = open(r'C:\Users\Administrator\Desktop\iris.txt','r')

    iris_data=np.empty((150,4),dtype=float)
    
    test_size=50-train_size
    train_num=3*train_size
    test_num=3*test_size
    a=0
    while True:
        line = f.readline()
        line_list=line.split(',')
        for i in range(4):
            if not line: break
            iris_data[a,i]=float(line_list[i])
            
        a+=1
        
        if not line: break

    f.close()


    iris_data0=iris_data[0:50,:]
    iris_data1=iris_data[50:100,:]
    iris_data2=iris_data[100:150,:]

    simple_num=range(0,50)
    random_index=random.sample(simple_num,50)
    train_index=random_index[0:train_size]
    test_index=random_index[train_size:50]
    print(train_index,test_index)

    #生成训练集
    train_data0=np.empty((train_size,4),dtype=float)
    train_data1=np.empty((train_size,4),dtype=float)
    train_data2=np.empty((train_size,4),dtype=float)

    train_lab0=np.zeros((train_size,3))
    train_lab1=np.zeros((train_size,3))
    train_lab2=np.zeros((train_size,3))
    train_lab0[:,0]=1
    train_lab1[:,1]=1
    train_lab2[:,2]=1
    train_lab = np.vstack((train_lab0,train_lab1,train_lab2))
    
    b=0
    for i in train_index:
        train_data0[b,:]=iris_data0[i,:]
        train_data1[b,:]=iris_data1[i,:]
        train_data2[b,:]=iris_data2[i,:]
        b+=1

    train_data = np.vstack((train_data0,train_data1,train_data2))

    #生成测试集
    test_data0=np.empty((test_size,4),dtype=float)
    test_data1=np.empty((test_size,4),dtype=float)
    test_data2=np.empty((test_size,4),dtype=float)

    test_lab0=np.zeros((test_size,3))
    test_lab1=np.zeros((test_size,3))
    test_lab2=np.zeros((test_size,3))
    test_lab0[:,0]=1
    test_lab1[:,1]=1
    test_lab2[:,2]=1
    test_lab = np.vstack((test_lab0,test_lab1,test_lab2))
    
    b=0
    for i in test_index:
        test_data0[b,:]=iris_data0[i,:]
        test_data1[b,:]=iris_data1[i,:]
        test_data2[b,:]=iris_data2[i,:]
        b+=1
    test_data = np.vstack((test_data0,test_data1,test_data2))

    return train_data,test_data,train_lab,test_lab

if __name__ =='__main__':
    read_iris(trian_size)

