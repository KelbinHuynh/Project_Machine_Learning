from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import imutils
import tensorflow as tf
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import joblib
with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.header('CHƯƠNG 9: K-NEAREST NEIGHBORS')
np.random.seed(100)

def bai1():
    centers = [[2, 3], [5, 5], [1, 8]]
    N = st.number_input("Nhập vào số dữ liệu",min_value = 4, step=1, format="%i")
    st.write("Cho 3 điểm nằm xung quanh")
    st.info(centers)
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)
    st.write("Chia " + str(N) +" điểm thành 3 nhóm")
    nhom_0 = []
    nhom_1 = []
    nhom_2 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])
        else:
            nhom_2.append([data[i,0], data[i,1]])
    
    nhom_0 = np.array(nhom_0)
    st.write("**Dữ liệu nhóm 0**")
    st.write(nhom_0.T)
    nhom_1 = np.array(nhom_1)
    st.write("**Dữ liệu nhóm 1**")
    st.write(nhom_1.T)
    nhom_2 = np.array(nhom_2)
    st.write("**Dữ liệu nhóm 2**")
    st.write(nhom_2.T)
    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)
    plt.plot(nhom_2[:,0], nhom_2[:,1], 'ob', markersize = 2)
    plt.legend(['Nhóm 0', 'Nhóm 1', 'Nhóm 2'])
    plt.show()  
    plt.title("K-nearest neighbor (k=3)")  
    st.pyplot(plt)
    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 
    # default k = n_neighbors = 5
    #         k = 3
    
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
 
    st.write("Sai số")
    st.info(sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = knn.predict(my_test)
   
   
    st.write("Kết quả nhận dạng là nhóm ")
    st.info(ket_qua[0])
   
def Bai02():
    # take the MNIST data and construct the training and testing split, using 75% of the
    # data for training and 25% for testing
    st.write("Lấy dữ liệu MNIST và xây dựng khóa đào tạo và thử nghiệm, sử dụng 75% "
     "dữ liệu để đào tạo và 25% để kiểm tra")
    mnist = datasets.load_digits()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
        mnist.target, test_size=0.25, random_state=42)
    st.write("Lấy 10% dữ liệu huấn luyện và sử dụng để xác thực ")
    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)
    st.write("Số điểm dữ liệu training")
    st.info(len(trainLabels))
    st.write("Số điểm dữ liệu validation")
    st.info(len(valLabels))
    st.write("Số điểm dữ liệu testing")
    st.info(len(testLabels))
    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    st.write("Đánh giá mô hình và cập nhật danh sách chính xác")
    st.info("%.2f%%" % (score * 100))
    st.write("Khởi tạo vòng lặp qua một vài chữ số ngẫu nhiên")
    # loop over a few random digits
    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
        st.write("1. Lấy hình ảnh và khởi tạo nó")
        # grab the image and classify it
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        image = image.reshape((8, 8)).astype("uint8")

        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        # show the prediction
        st.write("2. Hiển thị dự đoán. Tôi nghĩ chữ số đó là:")
        st.info("{}".format(prediction))
        st.image(image, clamp=True)

def Bai03():
    st.write("Lấy dữ liệu trên MNIST")
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    st.write("**X_train**")
    st.info(X_train)
    st.write("**Y_train**")
    st.info(Y_train)
    st.write("**X_test**")
    st.info(X_test)
    st.write("**Y_test**")
    st.info(Y_test)
    # 784 = 28x28
    RESHAPED = 784
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED) 
    st.write("**Lấy 10% dữ liệu training và sử dụng cho validation**")
    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
        test_size=0.1, random_state=84)

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)

    # save model, sau này ta sẽ load model để dùng 
    joblib.dump(model, "knn_mnist.pkl")
   
    # Đánh giá trên tập validation
    predicted = model.predict(valData)
    do_chinh_xac = accuracy_score(valLabels, predicted)
    st.write("Độ chính xác trên tập validation")
    st.info('%.0f%%' % (do_chinh_xac*100))
    # Đánh giá trên tập test
    predicted = model.predict(X_test)
    do_chinh_xac = accuracy_score(Y_test, predicted)
    st.write("Độ chính xác trên tập test")
    st.info('%.0f%%' % (do_chinh_xac*100))

def Bai03a():
    st.write("Lấy dữ liệu trên MNIST")
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    st.write("**X_train**")
    st.info(X_train)
    st.write("**Y_train**")
    st.info(Y_train)
    st.write("**X_test**")
    st.info(X_test)
    st.write("**Y_test**")
    st.info(Y_test)
    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]
    # 784 = 28x28
    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    knn = joblib.load("knn_mnist.pkl")
    predicted = knn.predict(sample)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            print('%2d' % (predicted[k]), end='')
            k = k + 1
        print()

    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1
    st.write("Hiển thị dự đoán")
    st.image(digit)
    cv2.waitKey(0)
page_names_to_funcs = {
    "Bài 1":bai1,
    "Bài 2":Bai02,
    "Bài 3": Bai03,
    "Bài 3a": Bai03a
}

demo_name = st.sidebar.selectbox("BÀI TẬP", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

