# %%
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pickle
from tqdm import tqdm

# %%
#function for 1 hot label encoding
def label_encoding(label):
    
    y = np.zeros([10,len(label)])
    
    for i in range(len(label)):
        y[int(label[i]),i] = 1
        
    return y

#function for computing classification performance
def performance_metrics(y_true,y_pred_ind):
    y_pred=np.zeros(y_true.shape)
    for i in range(len(y_pred_ind)):
        y_pred[i,y_pred_ind[i]]=1
    
    errors=np.zeros(y_pred.shape[1])
    accuracies=np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        errors[i]=np.sum(y_pred[:,i]!=y_true[:,i])/y_true.shape[0]
        accuracies[i]=1-errors[i]
    return accuracies,errors
    


#function for loading data
def load_data(f_loc, im_size):
    f_list = os.listdir(f_loc)
    x_data = np.zeros([len(f_list),im_size])

    for i in range(len(f_list)):
        im = mpimg.imread(f_loc + '/' + f_list[i])
        im = np.reshape(im,[1,im_size])
        x_data[i:i+1,0:im_size] = im
    
    x_data = x_data/255    #normalization
    x_data = np.float32(x_data)
    x_data = x_data.transpose()
    
    return x_data



# %%
#loading Train data
train_dir = r'train_data'
im_size = 784
train_data = load_data(train_dir, im_size)

#loading Test data
test_dir = r'test_data'
test_data = load_data(test_dir, im_size)




#loading and encoding Train Labels
path_to_label = r'labels'
tr_labels = np.loadtxt(path_to_label+'/'+'train_label.txt')
train_label = label_encoding(tr_labels)
train_label = np.float32(train_label)

#loading and encoding Test Labels
te_labels = np.loadtxt(path_to_label+'/'+'test_label.txt')
test_label = label_encoding(te_labels)
test_label = np.float32(test_label)

print("Train data shape:",train_data.shape)
print("Train label shape:",train_label.shape)
print("Test data shape:",test_data.shape)
print("Test label shape:",test_label.shape)

# %%
Y=np.transpose(train_label)

print(Y.shape)

X=np.transpose(train_data)

print(X.shape)

X_test=np.transpose(test_data)
print(X_test.shape)
Y_test=np.transpose(test_label)
print(Y_test.shape)

# %%


def regularized_gradient(w,mode=''): #Compute regularization gradient
    c=1e-6 #To prevent division by zero
    j=np.argmax(np.sum(np.abs(w),0)) #Getting Max L1 column
    b=np.zeros(w.shape)   

    if mode=='matrix':

        b[:,j]=np.round((w[:,j]+c)/(abs(w[:,j])+c)) #regularization gradient of matrix parameters

    elif mode=='vector':
        b=np.round((w+c)/(abs(w)+c)) #regularization gradient of vector parameters
    
    return b

def forward_propagation(X,Y,W1,W2,W3,b1,b2,b3,mode=0): #Forward Propagation for 2 hidden layer neural network 
    Z1 = tf.matmul(X,W1) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(A1,W2) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(A2,W3) + b3
    A3 = tf.nn.softmax(Z3)
    if mode==0:
        return Z1,Z2,Z3,A1,A2,A3
    else:
        return A3
    

def weight_update(x,Y,W1,W2,W3,b1,b2,b3,learning_rate,lam=1e-4,mode=0): #Function for weight update
    z1,z2,z3,h1,h2,out = forward_propagation(x,Y,W1,W2,W3,b1,b2,b3)
    
    loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(Y,tf.math.log(out)), axis=1))

    if mode==1:
        return loss
        
    ### backpropagation
    dy = -tf.divide(Y, out)   
    a= tf.expand_dims(out * dy, 1)
    b=tf.eye(k, batch_shape=[batch_size])
    c=tf.expand_dims(out,1)    
    dz3 = tf.squeeze(tf.matmul(a, b - c))
    a=tf.matmul(tf.expand_dims(h2,-1), tf.expand_dims(dz3, 1))
    dw3 = tf.reduce_mean(a, axis=0)
    db3 = tf.reduce_mean(dz3, axis=0)
    dh2 = tf.matmul(dz3, tf.transpose(w3))
    a=tf.matmul(tf.expand_dims(dh2, 1), tf.linalg.diag(tf.cast(z2 > 0, tf.float32)))
    dz2 = tf.squeeze(a)
    a=tf.matmul(tf.expand_dims(h1,-1), tf.expand_dims(dz2, 1))
    dw2 = tf.reduce_mean(a, axis=0)
    db2 = tf.reduce_mean(dz2, axis=0)
    dh1 = tf.matmul(dz2, tf.transpose(w2))
    a=tf.matmul(tf.expand_dims(dh1, 1), tf.linalg.diag(tf.cast(z1 > 0, tf.float32)))
    dz1 = tf.squeeze(a)
    a=tf.matmul(tf.expand_dims(x,-1), tf.expand_dims(dz1, 1))
    dw1 = tf.reduce_mean(a, axis=0)
    db1= tf.reduce_mean(dz1, axis=0)

    ### L1 regularization Gradient calculation
    dw3R = regularized_gradient(w3,'matrix')
    dw2R = regularized_gradient(w2,'matrix')
    dw1R = regularized_gradient(w1,'matrix')
    db3R = regularized_gradient(b3,'vector')
    db2R = regularized_gradient(b2,'vector')
    db1R = regularized_gradient(b1,'vector')


    #update weights
    W1 = W1 - learning_rate*(dw1+lam*dw1R)
    W2 = W2 - learning_rate*(dw2+lam*dw2R)
    W3 = W3 - learning_rate*(dw3+lam*dw3R)
    b1 = b1 - learning_rate*(db1+lam*db1R)
    b2 = b2 - learning_rate*(db2+lam*db2R)
    b3 = b3 - learning_rate*(db3+lam*db3R)

    return W1,W2,W3,b1,b2,b3,loss

# %%

#Hyperparameters
n_0 =784
num_classes = 10

n_1=100
n_2=100
k = Y.shape[1] #10
lr=0.05
lam=1e-4
batch_size=50
epoch=30
num_batches=int(len(X)/batch_size)
print(num_batches)




# %%
### Weight Initialization
tf.random.set_seed(0)
w1=tf.random.normal(    [n_0,n_1], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=None, name=None)

w2=tf.random.normal(    [n_1,n_2], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=None, name=None)

w3=tf.random.normal(   [n_2,k], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=None, name=None)

b1=tf.zeros(    [n_1], dtype=tf.dtypes.float32, name=None)

b2=tf.zeros(    [n_2], dtype=tf.dtypes.float32, name=None)

b3=tf.zeros(    [k], dtype=tf.dtypes.float32, name=None)


# %%
train_errors=[]
train_accuracy=[]
test_errors=[]
test_accuracy=[]
train_losses=[]
test_losses=[]


tr_loss=weight_update(X,Y,w1,w2,w3,b1,b2,b3,lr,mode=1)
te_loss=weight_update(X_test,Y_test,w1,w2,w3,b1,b2,b3,lr,mode=1)
train_losses.append(tr_loss)
test_losses.append(te_loss)


#TRAINING WITH SGD for defined number of EPOCHS

for i in tqdm(range(epoch)):
    index=list(range(X.shape[0]))
    random.shuffle(index) #shuffle the index to get random batches for iterations in each epoch. 

    #weight update over mini-batches    
    for j in range(num_batches): #Runs code over all batches in one epoch
        a=index[j*batch_size:(j+1)*batch_size]
        x=X[a,:]
        
        y=Y[a,:]
        
    
        w1,w2,w3,b1,b2,b3,cost=weight_update(x,y,w1,w2,w3,b1,b2,b3,lr,lam,mode=0)
    
    tr_loss=weight_update(X,Y,w1,w2,w3,b1,b2,b3,lr,mode=1)
    te_loss=weight_update(X_test,Y_test,w1,w2,w3,b1,b2,b3,lr,mode=1)
    train_losses.append(tr_loss)
    test_losses.append(te_loss)

   

    #Performance Evaluation after each epoch
    a3=forward_propagation(X,Y,w1,w2,w3,b1,b2,b3,mode=1)
    train_acc,train_error=performance_metrics(Y,np.argmax(a3,axis=1))

    
    a3test=forward_propagation(X_test,Y_test,w1,w2,w3,b1,b2,b3,mode=1)
    test_acc,test_error=performance_metrics(Y_test,np.argmax(a3test,axis=1))

    
    train_errors.append(train_error)
    train_accuracy.append(train_acc)
    test_errors.append(test_error)
    test_accuracy.append(test_acc)

# %%
#VISUALIZATION
train_errors=np.array(train_errors)
test_errors=np.array(test_errors)
train_accuracy=np.array(train_accuracy)
test_accuracy=np.array(test_accuracy)

for i in range(num_classes):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train_errors[:,i], label = "train error")
    plt.plot(test_errors[:,i], label = "test error")
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('Error for digit ' + str(i))
    plt.show()

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train_accuracy[:,i], label = "train accuracy")
    plt.plot(test_accuracy[:,i], label = "test accuracy")
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('Accuracy for digit ' + str(i))
    plt.show()

# %%
avg_er_train = np.sum(train_errors, 1)/num_classes
avg_er_test = np.sum(test_errors, 1)/num_classes
avg_acc_train = np.sum(train_accuracy, 1)/num_classes
avg_acc_test = np.sum(test_accuracy, 1)/num_classes

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(avg_er_train, label = "overall train error")
plt.plot(avg_er_test, label ="overall test error")
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('training and test error')
plt.show()

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(avg_acc_train, label = "overall train accuracy")
plt.plot(avg_acc_test, label ="overall test accuracy")
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('training and test accuracy')
plt.show()

# %%
#Average loss per epoch
train_losses=np.array(train_losses)
test_losses=np.array(test_losses)
train_losses_avg=train_losses/len(tr_labels)
test_losses_avg=test_losses/len(te_labels)


#Plot Losses
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(train_losses_avg, label = "train loss")
plt.plot(test_losses_avg, label = "test loss")
plt.xlabel('number of epochs')
plt.ylabel('training and test loss')
plt.legend()
plt.show()

# %%
#Checking and getting performance metrics
a3L=forward_propagation(X,Y,w1,w2,w3,b1,b2,b3,mode=1)
train_accL,train_errorL=performance_metrics(Y,np.argmax(a3L,axis=1))


a3testL=forward_propagation(X_test,Y_test,w1,w2,w3,b1,b2,b3,mode=1)
test_accL,test_errorL=performance_metrics(Y_test,np.argmax(a3testL,axis=1))


print("Training Error: ",train_errorL)
print("Test Error: ",test_errorL)
print("Training Accuracy: ",train_accL)
print("Test Accuracy: ",test_accL)

print('Average tr err',np.mean(train_errorL))
print('Average te err',np.mean(test_errorL))
print('Average tr acc',np.mean(train_accL))
print('Average te acc',np.mean(test_accL))

# %%
#savig the parameters as numpy arrays

theta=[w1.numpy(),b1.numpy(),w2.numpy(),b2.numpy(),w3.numpy(),b3.numpy()]
filehandler = open("nn_parameters.txt","wb")
pickle.dump(theta, filehandler, protocol=2)
filehandler.close()

# %%

#Verification that the parameters were saved correctly
file=open("nn_parameters.txt","rb")
theta_load=pickle.load(file)
file.close()
w1l=theta_load[0]
b1l=theta_load[1]
w2l=theta_load[2]
b2l=theta_load[3]
w3l=theta_load[4]
b3l=theta_load[5]

a3L=forward_propagation(X,Y,w1l,w2l,w3l,b1l,b2l,b3l,mode=1)
train_accL,train_errorL=performance_metrics(Y,np.argmax(a3L,axis=1))


a3testL=forward_propagation(X_test,Y_test,w1l,w2l,w3l,b1l,b2l,b3l,mode=1)
test_accL,test_errorL=performance_metrics(Y_test,np.argmax(a3testL,axis=1))


print("Training Error: ",train_errorL)
print("Test Error: ",test_errorL)
print("Training Accuracy: ",train_accL)
print("Test Accuracy: ",test_accL)

print('Average tr err',np.mean(train_errorL))
print('Average te err',np.mean(test_errorL))
print('Average tr acc',np.mean(train_accL))
print('Average te acc',np.mean(test_accL))



