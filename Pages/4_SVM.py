from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.header('CHƯƠNG 26. SUPPORT VECTOR MACHINE-SVM')
np.random.seed(100)
N = 150
def bai01():
    centers = [[2, 2], [7, 7]]
    st.write("Centers")
    st.info(centers)
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    st.write("Nhóm 0")
    st.info(nhom_0)
    st.write("Nhóm 1")
    st.info(nhom_1)

    svc = LinearSVC(C = 100, loss="hinge", random_state=42, max_iter = 100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write("Sai số")
    st.info(sai_so)
    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    st.write("Kết quả nhãn đăng là nhóm")
    st.info(ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')


    decision_function = svc.decision_function(train_data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = train_data[support_vector_indices]
    support_vectors_x = support_vectors[:,0]
    support_vectors_y = support_vectors[:,1]

    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        svc,
        train_data,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.legend(['Nhom 0', 'Nhom 1'])
    plt.title("Đồ thị")
    plt.show()    
    st.pyplot(plt)
def bai01a():
    centers = [[2, 2], [7, 7]]
    st.write("Center")
    st.info(centers)
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    st.write("Nhóm 0")
    st.info(nhom_0)
    st.write("Nhóm 1")
    st.info(nhom_1)
    svc = LinearSVC(C = 100, loss="hinge", random_state=42, max_iter = 100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write("Sai số")
    st.info(sai_so)
    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)
    st.write("Kết quả nhãn đăng là nhóm")
    st.info(ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')

    w = he_so[0]
    a = w[0]
    b = w[1]
    c = intercept[0]
    
    distance = np.zeros(SIZE, np.float32)
    for i in range(0, SIZE):
        x0 = train_data[i,0]
        y0 = train_data[i,1]
        d = np.abs(a*x0 + b*y0 + c)/np.sqrt(a**2 + b**2)
        distance[i] = d

    st.write("Khoảng cách")
    st.info(distance)
    vi_tri_min = np.argmin(distance)
    min_val = np.min(distance)
    st.write('Vị trí min')
    st.info(min_val)
 
    st.write("Những vị trí gần min")

    vi_tri = []
    for i in range(0, SIZE):
        if (distance[i] - min_val) <= 1.0E-3:
            print(distance[i])
            vi_tri.append(i)
    st.write(vi_tri)
    for i in vi_tri:
        x = train_data[i,0]
        y = train_data[i,1]
        plt.plot(x, y, 'rs')

    i = vi_tri[0]
    x0 = train_data[i,0]
    y0 = train_data[i,1]
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    i = vi_tri[2]
    x0 = train_data[i,0]
    y0 = train_data[i,1]
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')


    plt.legend(['Nhom 0', 'Nhom 1'])

    plt.show()    
    st.pyplot(plt)
def bai02():
    centers = [[2, 2], [7, 7]]
    st.write("Center")
    st.info(centers)
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)


    svc = SVC(C = 100, kernel='linear', random_state=42)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write("sai số")
    st.info(sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    st.write("Kết quả nhãn đăng là nhóm")
    st.info(ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]
    plt.plot(xx, yy, 'b')

    support_vectors = svc.support_vectors_
    st.write("Support vectors")
    st.info(support_vectors)

    w = he_so[0]
    a = w[0]
    b = w[1]

    i = 0
    x0 = support_vectors[i,0]
    y0 = support_vectors[i,1]
    plt.plot(x0, y0, 'rs')
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    i = 1
    x0 = support_vectors[i,0]
    y0 = support_vectors[i,1]
    plt.plot(x0, y0, 'rs')
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    plt.legend(['Nhom 0', 'Nhom 1'])
    plt.title("Đồ thị")
    plt.show()   
    st.pyplot(plt) 
def plot_linearsvc_support_vectors():
    st.write("**X,Y = make_blobs(n_samples=40, centers=2, random_state=0)**")
    X, y = make_blobs(n_samples=40, centers=2, random_state=0)
    st.write("**X**")
    st.info(X)
    st.write("**Y**")
    st.info(y)
    
    plt.figure(figsize=(10, 5))

    # "hinge" is the standard SVM loss
    clf = LinearSVC(C=100, loss="hinge", random_state=42).fit(X, y)
    # obtain the support vectors through the decision function
    decision_function = clf.decision_function(X)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    st.write("**NOTE**")
    st.write("1. Hinge là mất SVM tiêu chuẩn")
    st.write("2. Chúng ta cũng có thể tính toán các chức năng theo cách thủ công")
    st.write("3. Decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]")
    st.write("4. Các vector hỗ trợ là các mẫu nằm trong lề")
    st.write("5. Ranh giới có kích thước bị hạn chế theo quy ước")
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = X[support_vector_indices]

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title("C = 100")

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
page_names_to_funcs = {
    "Bài 01": bai01,
    "Bài 01a": bai01a,
    "Bài 02": bai02,
    "plot linearsvc support vectors": plot_linearsvc_support_vectors
}

demo_name = st.sidebar.selectbox("BÀI TẬP", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()