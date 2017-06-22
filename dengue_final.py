
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing, metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:

# Importing training csv files
feature_df = pd.read_csv("dengue_features_train.csv", index_col=[0,1,2])
label_df = pd.read_csv("dengue_labels_train.csv", index_col=[0,1,2])


# In[3]:

# Seperate data for San Juan
sj_train_features = feature_df.loc['sj']
sj_train_labels = label_df.loc['sj']

# Separate data for Iquitos
iq_train_features = feature_df.loc['iq']
iq_train_labels = label_df.loc['iq']


# In[4]:

# bar chart showing week of year to number of dengue cases relationship
label_df['mean_cases'] = label_df.groupby('weekofyear').total_cases.transform(np.mean)
mean_list = []
for index, case in enumerate(label_df['mean_cases'][35:87], 1):
    mean_list.append(case)

plt.bar(range(len(mean_list)), mean_list, align='center')
plt.xlabel('week of year')
plt.ylabel('dengue cases')

plt.show()


# In[5]:

# Remove 'week_start_date' string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)


# In[6]:

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases


# In[7]:

sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)


# In[8]:

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()


# In[9]:

# San Juan
(sj_correlations.total_cases.drop('total_cases').sort_values(ascending=False).plot.bar())
plt.show()


# In[10]:

# Iquitos
(iq_correlations.total_cases.drop('total_cases').sort_values(ascending=False).plot.bar())
plt.show()


# In[11]:

features = ['reanalysis_specific_humidity_g_per_kg', 
            'reanalysis_dew_point_temp_k', 
            'station_avg_temp_c', 
            'station_min_temp_c',
            'station_max_temp_c',
            'reanalysis_min_air_temp_k',
            'reanalysis_air_temp_k',
            'reanalysis_avg_temp_k',
            'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_relative_humidity_percent',
            "reanalysis_max_air_temp_k"
            ]


# In[12]:

# label dataframes for train
sj_train_label = sj_train_features.total_cases
iq_train_label = iq_train_features.total_cases


# In[13]:

# feature dataframes for train

sj_train_features = sj_train_features[features]
iq_train_features = iq_train_features[features]


# In[14]:

print('sj_train_label.shape: {}'.format(sj_train_label.shape))
print('iq_train_label.shape: {}'.format(iq_train_label.shape))
print()
print('sj_train_features.shape: {}'.format(sj_train_features.shape))
print('iq_train_features.shape: {}'.format(iq_train_features.shape))


# In[15]:

# Linear Regression for sj

x_df = sj_train_features
y_df = sj_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.LinearRegression(normalize=True)


reg.fit(x_train, y_train)

print('Linear regression for sj')


print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))


# In[16]:

# Linear Regression for iq

x_df = iq_train_features
y_df = iq_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.LinearRegression(normalize=True)


reg.fit(x_train, y_train)

print('Linear regression for iq')


print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))


# In[17]:

# Ridge regression for sj

x_df = sj_train_features
y_df = sj_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.RidgeCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=True).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Ridge regression for sj')


print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))



# In[18]:

# Ridge regression for iq

x_df = iq_train_features
y_df = iq_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.RidgeCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=False).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Ridge regression for iq')


print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))




# In[19]:

# Lasso regression for sj

x_df = sj_train_features
y_df = sj_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=True).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Lasso regression for sj')
print

print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))



# In[20]:

# Lasso regression for iq

x_df = iq_train_features
y_df = iq_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=False).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Lasso regression for iq')
print()

print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))



# In[21]:

# Elastic Net regression for sj

x_df = sj_train_features
y_df = sj_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.ElasticNetCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=False).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Elastic Net regression for sj')
print

print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))



# In[22]:

# Elastic Net regression for iq

x_df = iq_train_features
y_df = iq_train_label


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=4)

reg = linear_model.ElasticNetCV(alphas=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], normalize=False).fit(x_train,y_train)

reg.fit(x_train, y_train)

print('Elastic Net regression for iq')
print()

print("Score: ", reg.score(x_test, y_test))

predictions = reg.predict(x_test)

print("Mean absolute error:", eval_measures.meanabs(predictions, y_test))


# In[23]:

sj_train_frame = [sj_train_features, sj_train_label]
sj_train_merged = pd.concat(sj_train_frame, axis=1)

iq_train_frame = [iq_train_features, iq_train_label]
iq_train_merged = pd.concat(iq_train_frame, axis=1)


# In[24]:

# PCA(Principal Component Analysis) for SJ

X = sj_train_merged.ix[:,0:11].values
y = sj_train_merged.ix[:,11].values

X_std = preprocessing.StandardScaler().fit_transform(X)

# eigendecomposition on the covariance matrix

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

print("\n-------------\n")

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
count=0
print('Eigenvalues in descending order for SJ:')
for i in eig_pairs:
    print(count,i[0], list(sj_train_features)[count])
    count+=1


# In[25]:

# PCA(Principal Component Analysis) for IQ

X = iq_train_merged.ix[:,0:11].values
y = iq_train_merged.ix[:,11].values

X_std = preprocessing.StandardScaler().fit_transform(X)

# eigendecomposition on the covariance matrix

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

print("\n-------------\n")

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order for IQ:')
count=0
for i in eig_pairs:
    print(count,i[0], list(iq_train_features)[count])
    count+=1


# In[26]:

features = ["reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k", "station_avg_temp_c", "station_min_temp_c"]


# In[27]:

# Poisson Distribution
sj_train_subtrain, sj_train_subtest = train_test_split(sj_train_merged, test_size = 0.2)
iq_train_subtrain, iq_train_subtest = train_test_split(iq_train_merged, test_size = 0.2)

# We see that the top 4 features that PCA yields for SJ and IQ are the same, and hence choose them for features to use.
# Note that these are not identical to the feature-#cases correlations, as PCA detects correlations among variables 
# and cover more of the variance in the data.
formula_list = "total_cases ~ 1 + reanalysis_specific_humidity_g_per_kg + reanalysis_dew_point_temp_k + "                    "station_avg_temp_c + station_min_temp_c"


def p_rmae(train, test, formula, city):
    best_score_sj = 50
    best_score_iq = 10
    
    model = smf.glm(formula=formula, data=train, family=sm.families.Poisson())
    
    results = model.fit()
    predictions = results.predict(test).astype(int)
    score = eval_measures.meanabs(predictions, test.total_cases)
    
    if city == 'sj':
        if score < best_score_sj:
            best_score_sj = score
    else:
        if score < best_score_iq:
            best_score_iq = score
    
    if city == 'sj':
        print('mean absolute error for sj = ', best_score_sj)
    else:
        print('mean absolute error for iq = ', best_score_iq)
    
    

# sj_train_subtrain.shape
sj_result = p_rmae(sj_train_subtrain, sj_train_subtest, formula_list, 'sj')
iq_result = p_rmae(iq_train_subtrain, iq_train_subtest, formula_list, 'iq')


# In[28]:

# Negative Binomial

def nb_rmae(train, test, formula, city):
    best_score_sj = 50
    best_score_iq = 10
    
    model = smf.glm(formula=formula, data=train, family=sm.families.NegativeBinomial(alpha=0.01))
    
    results = model.fit()
    predictions = results.predict(test).astype(int)
    score = eval_measures.meanabs(predictions, test.total_cases)
    
    if city == 'sj':
        if score < best_score_sj:
            best_score_sj = score
    else:
        if score < best_score_iq:
            best_score_iq = score
    
    if city == 'sj':
        print('mean absolute error for sj = ', best_score_sj)
    else:
        print('mean absolute error for iq = ', best_score_iq)
    
    

# sj_train_subtrain.shape
sj_result = nb_rmae(sj_train_subtrain, sj_train_subtest, formula_list, 'sj')
iq_result = nb_rmae(iq_train_subtrain, iq_train_subtest, formula_list, 'iq')


# In[29]:

# Tensorflow multiple linear regression for SJ

x_df = sj_train_features
y_df = sj_train_label[:, np.newaxis]

X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.abs(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)

train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# for step in range(int(5e+5)+1):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_df, Y: y_df})
#     if step % 10000 == 0:
#         print(step, "Cost: ", cost_val)#, "\nPrediction:\n", hy_val)
'''
(0, 'Cost: ', 696.67584)
(10000, 'Cost: ', 241.54207)
(20000, 'Cost: ', 33.28595)
(30000, 'Cost: ', 33.048485)
(40000, 'Cost: ', 32.816826)
(50000, 'Cost: ', 32.587055)
'''


# In[30]:

# Tensorflow multiple linear regression for IQ

x_df = iq_train_features
y_df = iq_train_label[:, np.newaxis]


X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.abs(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)

train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# for step in range(int(5e+5)+1):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_df, Y: y_df})
#     if step % 10000 == 0:
#         print(step, "Cost: ", cost_val)#, "\nPrediction:\n", hy_val)

        
'''
(0, 'Cost: ', 862.51428)
(10000, 'Cost: ', 403.68274)
(20000, 'Cost: ', 66.840462)
(30000, 'Cost: ', 58.373543)
(40000, 'Cost: ', 57.234325)
(50000, 'Cost: ', 56.095345)
...
(450000, 'Cost: ', 35.078381)
(460000, 'Cost: ', 34.013134)
(470000, 'Cost: ', 32.950287)
(480000, 'Cost: ', 31.893694)
(490000, 'Cost: ', 30.837595)
(500000, 'Cost: ', 29.784908)

'''


# In[31]:

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)



def lstm(city):
    xy = pd.read_csv("dengue_features_train.csv", index_col=[0,1,2])
    xy_label = pd.read_csv("dengue_labels_train.csv", index_col=[0,1,2])

    if city == 'sj':
        sj_train_features = xy.loc['sj']
        sj_train_labels = xy_label.loc['sj']
        sj_train_features.drop('week_start_date', axis=1, inplace=True)
        sj_train_features.fillna(method='ffill', inplace=True)
        seq_length = 600
        xy = sj_train_features[::-1]
        xy = np.array(xy)
        xy = MinMaxScaler(xy)
        x = xy
        y = sj_train_labels["total_cases"]
        y = np.array(y)
    else:
        iq_train_features = xy.loc['iq']
        iq_train_labels = xy_label.loc['iq']
        iq_train_features.drop('week_start_date', axis=1, inplace=True)
        iq_train_features.fillna(method='ffill', inplace=True)
        seq_length = 300
        xy = iq_train_features[::-1]
        xy = np.array(xy)
        xy = MinMaxScaler(xy)
        x = xy
        y = iq_train_labels["total_cases"]
        y = np.array(y)


    data_dim = 20
    hidden_dim = 10
    output_dim = 1
    learning_rate = 1e-5
    iterations = 5

    # # build a dataset
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        dataX.append(_x)
        dataY.append(_y)


    # # train/test split
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])

    trainY=trainY[:, np.newaxis]
    testY=testY[:, np.newaxis]

    # # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)


    # cost/loss
    loss = tf.reduce_mean(tf.abs(Y_pred - Y))
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # Mean Absolute Error
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    mae = tf.reduce_mean(tf.abs(predictions - targets))


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    X: trainX, Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        mae = sess.run(mae, feed_dict={
                        targets: testY, predictions: test_predict})
        if city == 'sj':
            city = 'San Juan'
        else:
            city = "Iquitos"
        print("Mean Absolute Error for {}: {}".format(city, mae))

# lstm('sj')
# lstm('iq')
'''
[step: 0] loss: 17.998758316
[step: 1] loss: 17.9982280731
[step: 2] loss: 17.9976978302
[step: 3] loss: 17.997171402
[step: 4] loss: 17.9966430664
Mean Absolute Error for San Juan: 24.5960216522

[step: 0] loss: 9.44563674927
[step: 1] loss: 9.44528579712
[step: 2] loss: 9.44493293762
[step: 3] loss: 9.44458198547
[step: 4] loss: 9.44422912598
Mean Absolute Error for Iquitos: 4.86454200745
'''


# In[32]:

# Performances (measurement: Mean Absolute Error)

# San Juan:
    # RNN(LSTM): 24.59
    # Negative Binomial: 28.86
    # Poisson: 28.87
    # Elastic Net: 28.97
    # Lasso: 28.98
    # Ridge: 29.81
    # Tensorflow multiple linear regression: 32.58

# Iquitos
    # Negative Binomial: 4.69
    # Poisson: 4.69
    # RNN(LSTM): 4.86
    # Ridge: 5.72
    # Lasso: 5.87
    # Elastic Net: 5.89
    # Tensorflow multiple linear regression: 29.78

