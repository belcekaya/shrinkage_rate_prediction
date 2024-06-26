#################################################################
########   Correlation test to determine model variables ########
################################################################# 


# Step 1: loading Python libraries

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 

# Step 2: Computing Kendall correlogram
corr = df.corr(method='kendall') 

#define the threshold for high correlation
threshold = 0.70
print(np.where(corr > 0.70))

# Arrange your plot size according to your big data
f, ax = plt.subplots(figsize=(50, 50)) 

# Define your color palette for correlation matrix
cmap="RdYlBu" #you can also define different color palettes for your heatmap which are available in python like cmap="BrBG".

# Selecting palettes
pal = sns.color_palette("RdGy_r") # for Spearman correlogram reversed color palette
cmap2 =pal # for Spearman correlogram 

# Drawing the correlogram heatmap for Kendall correlogram

sns.heatmap(corr, cmap=cmap, vmax=0.7,
            center=0, annot=True,
            annot_kws={"size":10},
            square=True, linewidths=.05,
            linecolor='grey',
            cbar=True, mask=False, cbar_kws={"shrink":.5}, )

# Adding titles, adjusting subplots and labelling x/y ticks
plt.title('Kendall Correlogram', fontsize=13, fontfamily='serif')

# Arrange size of variables name headers 
plt.subplots_adjust(bottom=0.20,top=0.90, right=0.90, left=0.10)
plt.xticks(rotation=90, size=20)
plt.yticks(rotation=360, size=20)
plt.show() 

# Step 3: Computing Spearman correlogram

corr2 = df.corr(method='spearman') # for Spearman correlogram

#arrange your plot size according to your big data

f, ax = plt.subplots(figsize=(50, 50))

#define your color palette for correlation matrix

cmap="RdYlBu"

# Selecting palettes
pal = sns.color_palette("RdGy_r") # for Spearman correlogram reversed color palette
cmap2 =pal # for Spearman correlogram 

# Drawing the correlogram heatmap for spearman correlogram

sns.heatmap(corr2, cmap=cmap, vmax=0.7, center=0, annot=True,
            annot_kws={"size":30}, #size of the numbers inside of the correlation matrix cells
            square=True, linewidths=.05,
            linecolor='grey',
            cbar=True, mask=False, cbar_kws={"shrink":.5}, )

# Step 6: Adding titles, adjusting subplots and labelling x/y ticks

plt.title('Spearman correlogram', fontsize=50, fontfamily='serif')

plt.subplots_adjust(bottom=0.20,top=0.90, right=0.90, left=0.10)
plt.xticks(rotation=90, size=35)
plt.yticks(rotation=360, size=35)
plt.show() 


# # Step 4: Pearson correlogram with corrcoef function

x = np.array(df)
cor_mat = np.corrcoef(x.T)
cor_mat_df = pd.DataFrame(data = cor_mat, index=[df.columns], columns = df.columns) # for pearson

f, ax = plt.subplots(figsize=(50, 50))

cmap="RdYlBu"

# Step 5: Drawing the correlogramheatmap
sns.heatmap(cor_mat_df, cmap=cmap, vmax=0.7,
            center=0, annot=True,
            annot_kws={"size":30},
            square=True, linewidths=.05,
            linecolor='grey',
            cbar=True, mask=False, cbar_kws={"shrink":.5}, )
            
# Adding titles, adjusting subplots and labelling x/y ticks

plt.title('Pearson Correlogram',
fontsize=50, fontfamily='serif')

plt.subplots_adjust(bottom=0.20,top=0.90, right=0.90, left=0.10)
plt.xticks(rotation=90, size=35)
plt.yticks(rotation=360, size=35)
plt.show() 
