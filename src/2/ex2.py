# exercise 5.2.3

from pylab import *
import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Define names
names = ['word_freq_make', 'word_freq_address', 'word_freq_all',
         'word_freq_3d', 'word_freq_our', 'word_freq_over',
         'word_freq_remove', 'word_freq_internet', 'word_freq_order',
         'word_freq_mail', 'word_freq_receive', 'word_freq_will',
         'word_freq_people', 'word_freq_report', 'word_freq_addresses',
         'word_freq_free', 'word_freq_business', 'word_freq_email',
         'word_freq_you', 'word_freq_credit', 'word_freq_your',
         'word_freq_font', 'word_freq_000', 'word_freq_money',
         'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
         'word_freq_650', 'word_freq_lab', 'word_freq_labs',
         'word_freq_telnet', 'word_freq_857', 'word_freq_data',
         'word_freq_415', 'word_freq_85', 'word_freq_technology',
         'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
         'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
         'word_freq_original', 'word_freq_project', 'word_freq_re',
         'word_freq_edu', 'word_freq_table', 'word_freq_conference',
         'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
         'char_freq_$', 'char_freq_#']
tst=['capital_run_length_average', 'capital_run_length_longest',
     'capital_run_length_total']

#clean up names
names=[s.replace('word_freq_','').replace('char_freq_','') for s in names]

#Load data
freq = pd.read_csv("../data/spambase.data",names=names,usecols=range(54))
y = pd.read_csv("../data/spambase.data",usecols=[57],names=['spam'])

print(freq.head())

#Get data and standardize
df = pd.read_csv("../data/spambase.data",names=names+tst,usecols=range(57))
X = (df - df.mean()) / (df.max() - df.min())
X.head()

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


n_components = 54

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)


ax=pd.DataFrame(pca.explained_variance_).plot(title='PCA variance over number of PCA components')
ax.set_xlabel("PCA variance")
ax.set_ylabel("PCA components")


# Parameters
Kd = 5
Km = 5  # no of terms for regression model
N = 50  # no of data objects to train a model
Xe =  np.mat(linspace(-2,2,1000)).T # X values to visualize true data and model
eps_mean, eps_std = 0, 0.5          # noise parameters

# Generate dataset (with noise)
X = np.mat(linspace(-2,2,N)).T
Xd = np.power(X, range(1,Kd+1))
eps = np.mat(eps_std*np.random.randn(N) + eps_mean).T
w = np.mat( -np.power(-.9, range(1,Kd+2)) ).T
y = w[0,0] + Xd * w[1:,:] + eps



# Fit ordinary least squares regression model
Xm = np.power(X, range(1,Km+1))
model = lm.LinearRegression()
model = model.fit(Xm,y)

# Predict values
Xme = np.power(Xe, range(1,Km+1))
y_est = model.predict(Xme)

# Plot original data and the model output
f = figure()
f.hold(True)
plot(X.A,y.A,'.')
#plot(Xe.A,y_true.A,'-')
plot(Xe.A,y_est,'-')
xlabel('X'); ylabel('y'); ylim(-2,8)
legend(['Training data', 'Regression fit (model) K={0}'.format(Km)])

show()
