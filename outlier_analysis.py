#################################################################
#############          Detecting Outliers          ##############
#################################################################
df.shape
#(180, 42)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
print(IQR)

df = df[~((df < (Q1-1.5*IQR)) |(df > (Q3+1.5*IQR))).any(axis=1)]
df.shape
#(48, 42)  #132 rows are outliers.....

#z-score-outliers

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df))
print(z)
z_score = pd.DataFrame(data = z, index= range(180), columns = df.columns)

z_score.to_excel('C:\\Users\\XXX\\Desktop\\Z_score_outliers.xlsx')
threshold = 3
print(np.where(z > 3))
# print(z[18][21])
