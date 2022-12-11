from statistics import LinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes, load_boston
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint



rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


with open('./css/CaliHousing.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def Decision_Tree_Regression():
    st.header("Decision Tree Regression")
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.write('Mean:')
        st.info("%.2f" % scores.mean())
        st.write('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('Đăng tải file CSV tại đây'):
        uploaded_file = st.sidebar.file_uploader("upload", type=["csv"])
    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.write('**1. Dữ liệu về nhà ở**')

    if uploaded_file is not None:
        housing = pd.read_csv(uploaded_file)
        st.write(housing)
        housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_pipeline.fit_transform(housing)

    # Training
        st.write('**2. Huấn luyện**')
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)

    # Prediction
        st.write('**3. Dự đoán**')
        some_data = housing.iloc[:5]
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)
        some_labels = housing_labels.iloc[:5]
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)
        some_data_prepared = full_pipeline.transform(some_data)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)
        # Prediction 5 samples 
        st.write('**3.4 Dự đoán 5 mẫu**')
        st.write('Phỏng đoán')
        st.write(tree_reg.predict(some_data_prepared))
        st.write('Nhãn')
        st.info(list(some_labels))

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = tree_reg.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)

        st.write('**4. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)


    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu kiểm định chéo**')    
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = tree_reg.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**6. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')   
        st.info('%.2f' % rmse_test) 
    else:
        st.info('Đợi file CSV của bạn được load')
      
def Linear_Regression_UserModel():
    st.header("Linear regression userModl")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('Đăng tải dữ liệu của bạn'):
        uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])
    st.write('**1. Dữ liệu về nhà ở**')
    if uploaded_file is not None:
        housing = pd.read_csv(uploaded_file)
       
        st.write(housing)
        
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        # Load model lin_reg to use
        st.write("**2. Load model lin_reg để sử dụng**")
        lin_reg = LinearRegression()
        lin_reg = joblib.load("model_lin_reg.pkl")

        # Prediction
        st.write("**3. Phỏng đoán**")
        some_data = housing.iloc[:5]
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)

        some_labels = housing_labels.iloc[:5]
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)

        some_data_prepared = full_pipeline.transform(some_data)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)

        # Prediction 5 samples 
        st.write('**4. Dự đoán 5 mẫu**')
        st.write("Một vài dữ liệu chuẩn bị sẵn")
        st.write(lin_reg.predict(some_data_prepared))
        st.write('Một vài nhãn')
        st.info(list(some_labels))
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = lin_reg.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)
        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)


        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**6. Sai số bình phương trung bình trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = lin_reg.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)

        st.write('**7. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
    else:
        st.info('Đợi dữ liệu của bạn được load')
def Linear_Regression():
    st.header("Linear Regression")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]


    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('Đăng tải file CSV của bạn'):
        uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])
    st.write('**1. Dữ liệu về nhà ở**')

    if uploaded_file is not None:
        housing = pd.read_csv(uploaded_file)
        st.write(housing)
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)
        st.write("**2. Huấn luyện**")
        # Training
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        # Save model lin_reg 
        joblib.dump(lin_reg, "model_lin_reg.pkl")
        st.write("**3.Phỏng đoán**")
        # Prediction
        some_data = housing.iloc[:5]
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)

        some_labels = housing_labels.iloc[:5]
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)

        some_data_prepared = full_pipeline.transform(some_data)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)

        # Prediction 5 samples 
        st.write('**4. Dự đoán 5 mẫu**')
        st.write("Một vài dữ liệu chuẩn bị sẵn")
        st.write(lin_reg.predict(some_data_prepared))
        st.write('Một vài nhãn')
        st.write(list(some_labels))
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = lin_reg.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)
        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**6. Sai số bình phương trung bình trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = lin_reg.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**7. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
def Random_Forest_Regression_Grid_Search_CV():
    st.write("Random Forest Regression Grid Search CV")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('1. Đăng tải file CSV của bạn'):
        uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])
    st.write("**1. Dữ liệu về nhà ở**")
    if uploaded_file is not None:
        
        housing = pd.read_csv(uploaded_file)
        st.write(housing)
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                    ]
        # Training
        st.write("**2. Huấn luyện**")
        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                                scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)

        final_model = grid_search.best_estimator_

        # Prediction
        st.write("**3. Phỏng đoán**")

        some_data = housing.iloc[:5]
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)

        some_labels = housing_labels.iloc[:5]
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)

        some_data_prepared = full_pipeline.transform(some_data)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)
        
        # Prediction 5 samples 
        st.write('**4. Dự đoán 5 mẫu**')
        st.write("Một vài dữ liệu chuẩn bị sẵn")
        st.write(final_model.predict(some_data_prepared))
        st.write('Một vài nhãn')
        st.info(list(some_labels))

        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = final_model.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)

        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)
        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**6. Sai số bình phương trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = final_model.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**7. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
    else:
        st.info('Đợi dữ liệu của bạn được load')
def Random_Forest_Regression_Random_Search_CV_UseModel():
    st.header("Random Forest Regression Random Search CV useModel")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('1. Đăng tải file CSV của bạn'):
        uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])
    st.write("**1. Dữ liệu về nhà ở**")
    if uploaded_file is not None:
        
        housing = pd.read_csv(uploaded_file)
        st.write(housing)
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=1, high=8),
            }


        # Load model
        final_model = RandomForestRegressor()
        final_model = joblib.load("forest_reg_rand_search.pkl")
        st.write("**2. Load model**")

        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        # Prediction 5 samples 
        st.write("**3. Phỏng đoán**")
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)

        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = final_model.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)

        st.write('**4. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)
        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**5. Sai số bình phương trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)

        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = final_model.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**6. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
    else:
        st.info('Đợi dữ liệu của bạn được load')
def Random_Forest_Regression_Random_Search_CV():
    st.header("Random Forest Regression Random Search CV")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())

    with st.sidebar.header('Đăng tải dữ liệu của bạn'):
        uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])
    st.write("**1. Dữ liệu về nhà ở**")
    if uploaded_file is not None:
        housing = pd.read_csv(uploaded_file)
        st.write(housing)
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=1, high=8),
            }

        # Training
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(housing_prepared, housing_labels)

        final_model = rnd_search.best_estimator_
        joblib.dump(final_model, "forest_reg_rand_search.pkl")
        st.write("**2. Huấn luyện**")

        # Prediction
        st.write("**3. Phỏng đoán**")
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)

        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)

        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)

        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)

        # Prediction 5 samples 
        st.write('**4. Dự đoán 5 mẫu**')
        st.write("Một vài dữ liệu chuẩn bị sẵn")
        st.write(final_model.predict(some_data_prepared))
        st.write('Một vài nhãn')
        st.info(list(some_labels))
      
        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = final_model.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)

        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)
        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**6. Sai số bình phương trung bình trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = final_model.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**7. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
    else:
        st.info('Đợi dữ liệu của bạn được load')
def Random_Forest_Regression():
    st.header("Random Forest Regression")
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def display_scores(scores):
        st.markdown('Mean:')
        st.info("%.2f" % scores.mean())
        st.markdown('Standard deviation:')
        st.info("%.2f" % scores.std())
    with st.sidebar.header('Đăng tải file CSV của bạn'):
        uploaded_file = st.sidebar.file_uploader("upload", type=["csv"])
    st.write("**1. Dữ liệu về nhà ở**")
    if uploaded_file is not None:     
        housing = pd.read_csv(uploaded_file)
        # Them column income_cat dung de chia data
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
        st.write(housing)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Chia xong thi delete column income_cat
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

        # Training
        forest_reg = RandomForestRegressor()
        forest_reg.fit(housing_prepared, housing_labels)
        st.write("**2. Huấn luyện**")

        # Prediction
        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        st.write('**3.1 Dự đoán một vài dữ liệu**')
        st.write(some_data)
        st.write('**3.2 Dự đoán một vài nhãn**')
        st.write(some_labels)
        st.write('**3.3 Dự đoán một vài dữ liệu chuẩn bị sẵn**')
        st.write(some_data_prepared)
        # Prediction 5 samples 

        st.write('**4. Dự đoán 5 mẫu**')
        st.write("Một vài dữ liệu chuẩn bị sẵn")
        st.write(forest_reg.predict(some_data_prepared))
        st.write('Một vài nhãn')
        st.info(list(some_labels))

        # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
        housing_predictions = forest_reg.predict(housing_prepared)
        mse_train = mean_squared_error(housing_labels, housing_predictions)
        rmse_train = np.sqrt(mse_train)
        st.write('**5. Sai số bình phương trung bình trên tập dữ liệu huấn luyện**')
        st.info('%.2f' % rmse_train)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
        scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

        st.write('**6. Sai số bình phương trên tập dữ liệu kiểm định chéo**')
        rmse_cross_validation = np.sqrt(-scores)
        display_scores(rmse_cross_validation)

        # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        y_predictions = forest_reg.predict(X_test_prepared)

        mse_test = mean_squared_error(y_test, y_predictions)
        rmse_test = np.sqrt(mse_test)
        st.write('**7. Sai số bình phương trung bình trên tập dữ liệu kiểm tra**')
        st.info('%.2f' % rmse_test)
    else:
        st.info('Đợi dữ liệu của bạn được load')

page_names_to_funcs = {
    "Linear Regression":Linear_Regression,
    "Liner Regression UserModel":Linear_Regression_UserModel,
    "Decision Tree Regression": Decision_Tree_Regression,
    "Random Forest Regression":Random_Forest_Regression,
    "Random Forest Regression Random Search CV":Random_Forest_Regression_Random_Search_CV,
    "Random Forest Regression Random Search CV UseModel":Random_Forest_Regression_Random_Search_CV_UseModel,
    "Random Forest Regression Grid Search CV":Random_Forest_Regression_Grid_Search_CV,
}

demo_name = st.sidebar.selectbox("BÀI TẬP", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()