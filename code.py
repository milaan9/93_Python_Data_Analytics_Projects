import pandas_datareader as web
import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


from sklearn.model_selection import train_test_split

plt.style.use('bmh')

start = datetime.datetime(2010,1,1)
end = datetime.datetime.now()
with st.sidebar:
    selected = option_menu(
        menu_title=" Main Menu ",
        options = ["CLASSIFICATION", "LINEAR REGRESSION", "SUPPORT VECTOR REGRESSION", "DECISION TREE REGRESSION", "RANDOM FOREST REGRESSION"],
    )
if selected == "CLASSIFICATION":
    st.title("STOCK ANALYSIS")
    user_input = st.text_input("ENTER YOUR STOCK", 'NFLX')
    df = web.DataReader(user_input, 'yahoo', start, end)

    st.subheader("Data")
    st.write(df.describe())

    st.subheader("CLOSING PRICE V/S TIME", 'r')
    fig = plt.figure(figsize = (18,6))
    plt.plot(df.Close,'g')
    st.pyplot(fig)

    st.subheader("100 DAYS MOVING AVERAGE", 'r')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (18,6))
    plt.plot(ma100,'r')
    plt.plot(df.Close,'g')
    st.pyplot(fig)
    st.subheader("Returns")
    df['Return'] = df['Adj Close'].pct_change(365).shift(-365)
    list_of_features = ['High', 'Low', 'Open', 'Close','Volume','Adj Close']
    X = df[list_of_features]
    X.tail()
    y= np.where(df.Return > 0,1,0)
    df
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state= 423)
    treeClassifier = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 6)
    treeClassifier.fit(X_train,y_train)
    y_pred = treeClassifier.predict(X_test)
    st.subheader("WILL THIS STOCK GENERATE A RETURN?")
    from sklearn.metrics import classification_report
    report = classification_report(y_test,y_pred)
    st.write(report)
if selected == "DECISION TREE REGRESSION":
    st.title("STOCK ANALYSIS")
    user_input = st.text_input("ENTER YOUR STOCK", 'NFLX')
    df = web.DataReader(user_input, 'yahoo', start, end)

    st.subheader("Data")
    st.write(df.describe())

    #visualization
    st.subheader("CLOSING PRICE V/S TIME")
    fig = plt.figure(figsize=(18, 6))
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    st.subheader("100 DAYS MOVING AVERAGE", 'r')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(18, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

     #head
    st.subheader("SAMPLE")
    st.write("HEAD")
    future_days = 365
    df['Prediction'] = df[['Adj Close']].shift(-future_days)
    st.write(df.head())
    #tail
    st.write("TAIL")
    st.write(df.tail())

    #dependent
    X = np.array(df.drop(['Prediction'], 1))[:-future_days]

    #independent
    y = np.array(df['Prediction'])[:-future_days]
    #training
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #model = DecisionTreeRegressor()
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)
    x_future = df.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    #result
    st.subheader("PREDICTED VALUES")
    tree_prediction = tree.predict(x_future)
    print(tree_prediction)
    lr_prediction = lr.predict(x_future)
    st.write(lr_prediction)

    #graphical_visualization
    st.subheader("PREDICTION GRAPH")
    predictions = tree_prediction
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig = plt.figure(figsize=(16, 8))
    plt.title('Decision Tree Regression')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Orignal', 'Val', 'Predicted'])
    st.pyplot(fig)
if selected == "LINEAR REGRESSION":
    st.title("STOCK ANALYSIS")
    user_input = st.text_input("ENTER YOUR STOCK", 'NFLX')
    df = web.DataReader(user_input, 'yahoo', start, end)

    st.subheader("Data")
    st.write(df.describe())

    st.subheader("SAMPLE")
    st.write("HEAD")
    forecast = 365
    df['Prediction'] = df[['Adj Close']].shift(-forecast)
    st.write(df.head())

    # tail
    st.write("TAIL")
    st.write(df.tail())

    #visualization
    st.subheader("CLOSING PRICE V/S TIME", 'r')
    fig = plt.figure(figsize=(18, 6))
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    st.subheader("100 DAYS MOVING AVERAGE", 'r')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(18, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    #X
    X = np.array(df.drop(['Prediction'], 1))
    X = X[:-forecast]
    #y
    y = np.array(df['Prediction'])
    y = y[:-forecast]
    #split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    linear_regression_confidence = linear_regression.score(x_test, y_test)
    st.write("Linear Regression confidence= ", linear_regression_confidence)

    x_forecast_out = np.array(df.drop(['Prediction'], 1))[-forecast:]

    linear_regression_prediction = linear_regression.predict(x_forecast_out)
    st.write(linear_regression_prediction)
    st.subheader("PREDICTION GRAPH")
    predictions = linear_regression_prediction
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig2 = plt.figure(figsize=(16, 8))
    plt.title('Linear Regression Prediction')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.plot(df['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Orignal', 'Val', 'Predicted'])
    st.pyplot(fig2)
if selected == "SUPPORT VECTOR REGRESSION":
    st.title("STOCK ANALYSIS")
    user_input = st.text_input("ENTER YOUR STOCK", 'NFLX')
    df = web.DataReader(user_input, 'yahoo', start, end)

    st.subheader("Data")
    st.write(df.describe())

    st.subheader("SAMPLE")
    st.write("HEAD")
    forecast = 365
    df['Prediction'] = df[['Adj Close']].shift(-forecast)
    st.write(df.head())

    # tail
    st.write("TAIL")
    st.write(df.tail())
    # visualization
    st.subheader("CLOSING PRICE V/S TIME", 'r')
    fig = plt.figure(figsize=(18, 6))
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    st.subheader("100 DAYS MOVING AVERAGE", 'r')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(18, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'g')
    st.pyplot(fig)


    # X
    X = np.array(df.drop(['Prediction'], 1))
    X = X[:-forecast]
    # y
    y = np.array(df['Prediction'])
    y = y[:-forecast]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    svm_confidence = svr_rbf.score(x_test, y_test)
    st.write('SVM confidence: ', svm_confidence)

    x_forecast_out = np.array(df.drop(['Prediction'], 1))[-forecast:]

    svm_prediction = svr_rbf.predict(x_forecast_out)
    st.write(svm_prediction)

    st.subheader("PREDICTION GRAPH")
    predictions = svm_prediction
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig3 = plt.figure(figsize=(16, 8))
    plt.title('SUPPORT VECTOR REGRESSION')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.plot(df['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Orignal', 'Val', 'Predicted'])
    st.pyplot(fig3)
if selected == 'RANDOM FOREST REGRESSION':
    st.title("STOCK ANALYSIS")
    user_input = st.text_input("ENTER YOUR STOCK", 'NFLX')
    df = web.DataReader(user_input, 'yahoo', start, end)

    st.subheader("Data")
    st.write(df.describe())

    st.write("HEAD")
    forecast = 365
    df['Prediction'] = df[['Adj Close']].shift(-forecast)
    st.write(df.head())

    # tail
    st.write("TAIL")
    st.write(df.tail())

    st.subheader("CLOSING PRICE V/S TIME", 'r')
    fig = plt.figure(figsize=(18, 6))
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    st.subheader("100 DAYS MOVING AVERAGE", 'r')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(18, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'g')
    st.pyplot(fig)

    future_days = 365
    df['Prediction'] = df[['Close']].shift(-future_days)
    df.tail(4)

    #X
    X = np.array(df.drop(['Prediction'], 1))[:-future_days]
    #Y
    y = np.array(df['Prediction'])[:-future_days]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    Etree = RandomForestRegressor().fit(x_train, y_train)

    x_E_future = df.drop(['Prediction'], 1)[:-future_days]
    x_E_future = x_E_future.tail(future_days)
    x_E_future = np.array(x_E_future)

    st.subheader("PREDICTED VALUES")
    tree_E_prediction = Etree.predict(x_E_future)
    st.write(tree_E_prediction)

    st.subheader("PREDICTION GRAPH")
    predictions = tree_E_prediction
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig4 = plt.figure(figsize=(16, 8))
    plt.title('Random Forest Regression')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Orignal', 'Val', 'Predicted'])
    st.pyplot(fig4)