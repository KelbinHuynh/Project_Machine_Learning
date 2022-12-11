import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.header("CHƯƠNG 8. OVERFITTING")
def bai01a():
    np.random.seed(100)
    N = st.number_input("Nhập vào số dữ liệu",min_value = 1, step=1, format="%i")
    X = np.random.rand(int(N), 1)*5
    st.write("X = np.random.rand(N, 1)*5")
    st.info(X)
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
    st.write("y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)")
    st.info(y)
    poly_features = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    N_test = 20 

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias

    lin_reg.fit(X_poly, y)


    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)


    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình - tập training')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))
    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
    st.write('Sai số bình phương trung bình - tập test')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))

    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 2')
    plt.show()
    st.pyplot(plt)
def bai01b():
    np.random.seed(100)

    N = st.number_input("Nhập vào số dữ liệu",min_value = 1, step=1, format="%i")
    X = np.random.rand(int(N), 1)*5
    st.write("X = np.random.rand(N, 1)*5")
    st.info(X)
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
    st.write("y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)")
    st.info(y)
    poly_features = PolynomialFeatures(degree=4, include_bias=True)
    X_poly = poly_features.fit_transform(X)


    N_test = 20 

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias

    lin_reg.fit(X_poly, y)


    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình - tập training')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))
    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
    st.write('Sai số bình phương trung bình - tập test')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))

    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hoi quy da thuc bac 4')

    plt.show()
    st.pyplot(plt)
def bai01c():
    np.random.seed(100)

    N = st.number_input("Nhập vào số dữ liệu",min_value = 1, step=1, format="%i")

    X = np.random.rand(int(N), 1)*5
    st.write("X = np.random.rand(N, 1)*5")
    st.info(X)
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
    st.write("y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)")
    st.info(y)
    poly_features = PolynomialFeatures(degree=8, include_bias=True)
    X_poly = poly_features.fit_transform(X)


    N_test = 20 

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias

    lin_reg.fit(X_poly, y)


    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)




    print(np.min(y_test), np.max(y) + 100)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình - tập training')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))

    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
    st.write('Sai số bình phương trung bình - tập test')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))


    plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])




    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 8')

    plt.show()
    st.pyplot(plt)
def bai01d():
    np.random.seed(100)
    N = st.number_input("Nhập vào số dữ liệu",min_value = 1, step=1, format="%i")
    X = np.random.rand(int(N), 1)*5
    st.write("X = np.random.rand(N, 1)*5")
    st.info(X)
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
    st.write("y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)")
    st.info(y)
    poly_features = PolynomialFeatures(degree=8, include_bias=True)
    X_poly = poly_features.fit_transform(X)


    N_test = 20 

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias

    lin_reg.fit(X_poly, y)


    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)

    print(np.min(y_test), np.max(y) + 100)

    plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình - tập training')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))
    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
    st.write('Sai số bình phương trung bình - tập test')
    st.info('%.6f' % (sai_so_binh_phuong_trung_binh/2))



    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 16')

    plt.show()
    st.pyplot(plt)
page_names_to_funcs = {
    "Bài 1a":bai01a,
    "Bài 1b":bai01b,
    "Bài 1c": bai01c,
    "Bài 1d": bai01d
    }

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


