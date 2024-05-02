import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import learning_curve
sorted(sklearn.metrics.SCORERS.keys())

# Set different train sizes
train_sizes = [1, 2, 5, 9,13, 25, 36, 72,90, 108, 126, 135, 144]
# (180*0.50 - 180*0.80)

# Prepare Feature set X and Target Y
CAO = df.iloc[:,-1].values
Cikis_AO = pd.DataFrame(data = CAO, index= range(180), columns = ['CAO'])

df  = df.iloc[:,0:33]
features = df.columns
X = df[features] #feature definition

# Standardize the Feature Set
sc= StandardScaler()

df_scaled = sc.fit_transform(df)
df = pd.DataFrame(data=df_scaled, index = range(180), columns = [X.columns])

Cikis_AO = sc.fit_transform(Cikis_AO)
Cikis_AO = pd.DataFrame(data=Cikis_AO, index = range(180), columns = ['CAO'])


# 1. SVR Training 

features = df.columns
target = 'CAO'

train_sizes, train_scores, validation_scores = learning_curve(estimator = SVR(),
                                                              X = df[features],
                                                              y = Cikis_AO[target], 
                                                              train_sizes = train_sizes, 
                                                              cv = 10,
                                                              scoring = 'neg_mean_squared_error', 
                                                              shuffle=True)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = np.abs(train_scores).mean(axis = 1)
validation_scores_mean = np.abs(validation_scores).mean(axis =1 )

print('Mean training scores\n\n', pd.Series(train_scores_mean, 
                                            index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, 
                                               index = train_sizes))


plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a SVR', fontsize = 18 , y = 1.03)
plt.legend()
plt.ylim(-3,3)

# 2. GBM Training

features = df.columns
target = 'CAO'
train_sizes, train_scores, validation_scores = learning_curve(estimator = GradientBoostingRegressor(),
                                                              X = df[features],
                                                              y = Cikis_AO[target], 
                                                              train_sizes = train_sizes, 
                                                              cv = 10,
                                                              scoring = 'neg_mean_squared_error', 
                                                              shuffle=True)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = np.abs(train_scores).mean(axis = 1)
validation_scores_mean = np.abs(validation_scores).mean(axis =1 )

print('Mean training scores\n\n', pd.Series(train_scores_mean, 
                                            index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, 
                                               index = train_sizes))

# Plot the training curve
plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a GBM', fontsize = 18 , y = 1.03)
plt.legend()
plt.ylim(-3,3)
