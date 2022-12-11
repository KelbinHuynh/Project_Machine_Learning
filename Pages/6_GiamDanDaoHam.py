from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.header("Giảm dần đạo hàm")
def bai01():
    def grad(x):
        return 2*x+ 5*np.cos(x)
    def cost(x):
        return x**2 + 5*np.sin(x)

    def myGD1(x0, eta):
        x = [x0]
        for it in range(100):
            x_new = x[-1] - eta*grad(x[-1])
            if abs(grad(x_new)) < 1e-3: # just a small number
                break
            x.append(x_new)
        return (x, it)

    if __name__ == "__main__":
        x0 = -5
        eta = 0.1
        (x, it) = myGD1(x0, eta)
        x = np.array(x)
        y = cost(x)
        n = 101
        xx = np.linspace(-6, 6, n)
        yy = xx**2 + 5*np.sin(xx)
        st.write("**Mảng xx**")
        st.info(xx)
        st.write("**Mảng yy** (đạo hàm của xx)")
        st.info(yy.T)
        plt.subplot(2,4,1)
        plt.plot(xx, yy)
   
        index = 0
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])
        plt.subplot(2,4,2)
        plt.plot(xx, yy)

        index = 1
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,3)
        plt.plot(xx, yy)
        index = 2
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,4)
        plt.plot(xx, yy)
        index = 3
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,5)
        plt.plot(xx, yy)
        index = 4
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,6)
        plt.plot(xx, yy)
        index = 5
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,7)
        plt.plot(xx, yy)
        index = 7
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.subplot(2,4,8)
        plt.plot(xx, yy)
        index = 11
        plt.plot(x[index], y[index], 'ro')
        s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
        plt.xlabel(s, fontsize = 8)
        plt.axis([-7, 7, -10, 50])

        plt.tight_layout()
        plt.show()
    
        st.pyplot(plt)
        st.write("**Nghiệm tìm được qua các vòng lặp**")
def bai02():
    st.write("Tạo mảng x=np.random.rand(1000)")
    X = np.random.rand(1000)
    st.info(X)
    st.write("Tạo mảng y=4 + 3 * X + .5*np.random.randn(1000)")
    y = 4 + 3 * X + .5*np.random.randn(1000)
    st.info(y)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    x0 = 0
    x1 = 1
    y0 = w*x0 + b
    y1 = w*x1 + b

    plt.plot(X, y, 'bo', markersize = 2)
    plt.plot([x0, x1], [y0, y1], 'r')
    st.write("Nghiệm của bài toán linear regression tìm được bằng thư viện scikit-learn")
    plt.show()
    st.pyplot(plt)
def bai02a():
    st.write("Tạo mảng x=np.random.rand(1000)")
    X = np.random.rand(1000)
    st.write("Tạo mảng y=4 + 3 * X + .5*np.random.randn(1000)")
    y = 4 + 3 * X + .5*np.random.randn(1000) # noise added

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    w, b = model.coef_[0][0], model.intercept_[0]
    sol_sklearn = np.array([b, w])
    print('Solution found by sklearn:', sol_sklearn)
    st.write("Giải pháp được tìm thấy bởi sklearn", sol_sklearn)
    # Building Xbar 
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    st.write('Giải pháp được tìm bởi GD: w = ', w1[-1], ',\n sau %d lần lặp.' %(it1+1))
def bai03():
    np.random.seed(100)
    N = 1000
    st.write("Tạo mảng X lưu trữ random 1000 điểm")
    X = np.random.rand(N)
    st.write("Tạo mảng  y = 4 + 3 * X + .5*np.random.randn(N) từ X")
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    print('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    for item in w1:
        print(item, cost(item))

    print(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    temp = w1[0]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax = plt.axes(projection="3d")
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[1]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[2]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[3]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)


    ax.plot_wireframe(b, w, z)
    ax.set_xlabel("b")
    ax.set_ylabel("w")
    plt.show()
    st.pyplot(plt)
def bai04():
    x = np.linspace(-2, 2, 21)
    st.write('x = np.linspace(-2, 2, 21)')
    st.info(x)
    y = np.linspace(-2, 2, 21)
    st.write('y = np.linspace(-2, 2, 21)')
    st.info(y)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    st.write('Z = X.X + Y.Y')
    st.info(Z)
    plt.contour(X, Y, Z, 10)
    plt.show()
    st.write("Đồ thị của X, Y và Z")
    st.pyplot(plt)
def bai05():
    np.random.seed(100)
    N = 1000
    st.write("X = np.random.rand(N)")
    X = np.random.rand(N)
    st.info(X)
    y = 4 + 3 * X + .5*np.random.randn(N)
    st.write("y = 4 + 3 * X + .5*np.random.randn(N)")
    st.info(y)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    print('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    st.write('Nghiệm tìm được bởi GD: w = ', w1[-1], ',\n sau %d lặp lại.' %(it1+1))
    for item in w1:
        print(item, cost(item))

    print(len(w1))
    st.write(w1)
    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    plt.contour(b, w, z, 45)
    bdata = []
    wdata = []
    for item in w1:
        plt.plot(item[0], item[1], 'ro', markersize = 3)
        bdata.append(item[0])
        wdata.append(item[1])

    plt.plot(bdata, wdata, color = 'b')

    plt.xlabel('b')
    plt.ylabel('w')
    plt.axis('square')
    plt.show()
    plt.title("Nghiệm của bài toán linear regression")
    st.pyplot(plt)
def temp():
    ax = plt.axes(projection="3d")
    st.write("X = np.linspace(-2, 2, 21)")
    X = np.linspace(-2, 2, 21)
    st.info(X)
    st.write("Y = np.linspace(-2, 2, 21)")
    Y = np.linspace(-2, 2, 21)
    st.info(Y)
    X, Y = np.meshgrid(X, Y)
    st.write("Z = X.X + Y.Y")
    Z = X*X + Y*Y
    st.info(Z)
    plt.title("Đồ thị không gian X,Y,Z")
    ax.plot_wireframe(X, Y, Z)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    st.pyplot(plt)

page_names_to_funcs = {
    "Bài 01": bai01,
    "Bài 02": bai02,
    "Bài 02a": bai02a,
    "Bài 03": bai03,
    "Bài 04": bai04,
    "Bài 05": bai05,
    "Temp": temp
}

demo_name = st.sidebar.selectbox("BÀI TẬP", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


