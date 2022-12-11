import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import streamlit as st
from sklearn.metrics import mean_squared_error

with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.header('CHƯƠNG 7: HỒI QUY TUYẾN TÍNH')

# height (cm), input data, each row is a data point
def bai1():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    st.write("Tạo một **mảng X(cm)** chứa giá trị chiều cao")
    st.write(X)
    one = np.ones((1, X.shape[1]))
    Xbar = np.concatenate((one, X), axis = 0) # each point is one row
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    st.write("Tạo **vector cột Y(kg)** chứa cân nặng ")
    st.write(y.T)

    A = np.dot(Xbar, Xbar.T)
    b = np.dot(Xbar, y)
    w = np.dot(np.linalg.pinv(A), b)
    # weights
    heightinput = st.slider('Chọn chiều cao:', 147, 183, 147)
        
    w_0, w_1 = w[0], w[1]
    y1 = w_1*heightinput + w_0
    st.write("Cho chiều cao là " + str(heightinput) + "(kg). Cân nặng dự đoán là:")
    st.info('%.2f'%(y1))

def bai2():
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    st.write("Tạo một **mảng X(cm)** chứa giá trị chiều cao")
    st.write(X.T)
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    st.write("Tạo một **mảng Y(kg)** chứa cân nặng")
    st.write(y.T)
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    X = X[:,0]
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    plt.show()

    plt.title("Đồ thị dự đoán chiều cao và cân nặng")
    st.pyplot(plt)
 
def bai3():
    st.write("Cho một số nguyên **m** có giá trị bằng")
    m = st.number_input("Nhập giá trị số nguyên m",min_value = 1, step=1, format="%i")
    st.write("Mảng **X = 6 * random(m,1) - 3**")
    X = 6 * np.random.rand(int(m), 1) - 3
    st.write(X.T)
    st.write("Một mảng **y = 0.5*X^2 + X + 2 + random(m,1)**")
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    st.write(y.T)
    X2 = X**2
   
    # print(X)
    # print(X2)
    
    X_poly = np.hstack((X, X2))
    # print(X_poly)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_)
    print(lin_reg.coef_)
    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0,0]
    c = lin_reg.coef_[0,1]
    print(a)
    print(b)
    print(c)

    x_ve = np.linspace(-3,3,m)
    y_ve = a + b*x_ve + c*x_ve**2

    plt.plot(X, y, 'o')
    plt.plot(x_ve, y_ve, 'r')
    # Tinh sai so
    loss = 0 
    for i in range(0, m):
        y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
        sai_so = (y[i] - y_mu)**2 
        loss = loss + sai_so
    loss = loss/(2*m)
    st.write("Sai số")
    st.info('%.6f' % loss)
    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write("Sai số bình phương trung bình")
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))
    plt.title("Đồ thị dự đoán")
    plt.show()
    st.pyplot(plt)
def bai4():
    # height (cm), input data, each row is a data point
    st.write("Tạo một **mảng X(cm)** chứa giá trị chiều cao")
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    st.write(X.T)
    st.write("Tạo một **mảng Y(kg)** chứa cân nặng")
    y = np.array([[ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    st.write(y.T)
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("Giải pháp scikit-kearn's")
    st.write("a. w_1 ")
    st.info(regr.coef_[0])
    st.write("b. w_0")
    st.info(regr.intercept_)

    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]

    plt.plot(x, y)
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")
    plt.show()
    plt.title("Đồ thị hiển thị chiều cao theo cân nặng")
    st.pyplot(plt)
def bai5():
    # height (cm), input data, each row is a data point
    st.write("Tạo một **mảng X(cm)** chứa giá trị chiều cao")
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    st.write(X.T)
    st.write("Tạo một **mảng Y(kg)** chứa cân nặng")
    y = np.array([[ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    st.write(y.T)
    huber_reg = linear_model.HuberRegressor()
    huber_reg.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("**Giải pháp scikit-kearn's**")
    st.write("a. w_1 ")
    st.info(huber_reg.coef_[0])
    st.write("b. w_0")
    st.info(huber_reg.intercept_)
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = huber_reg.coef_[0]
    b = huber_reg.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")
    plt.title("Đồ thị")
    plt.plot(x, y)
    plt.show()
    st.pyplot(plt)
page_names_to_funcs = {
    "Bài 1":bai1,
    "Bài 2": bai2, 
    "Bài 3": bai3,
    "Bài 4": bai4,
    "Bài 5": bai5
}

demo_name = st.sidebar.selectbox("BÀI TẬP", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()