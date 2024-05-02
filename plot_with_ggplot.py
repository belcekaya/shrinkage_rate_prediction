from plotnine import *
from plotnine.data import *
%matplotlib inline

# Scatter Plot
x =  label_test # you just need to define your actual test sample
y =  ypred # predicted values of output

(
    ggplot(modeltestsample, aes(x, y))
    + geom_point(alpha=0.5,size=3)
    + geom_smooth(colour='navy')
    + ggtitle("Model Performance Graph-XGBoost (Test set)")
    + xlab('CAO')
    + ylab('Pred_CAO')
    + geom_abline(aes(slope=1,intercept=0),colour='red')
)
