import numpy as np
import pandas as pd
from  random import  randrange
from sklearn import  datasets
import  lightgbm as lgb
import  gc
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import KFold,StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

train=pd.read_csv('../first_round_training_data.csv')
test=pd.read_csv('../first_round_testing_data.csv')
submit=pd.read_csv('../submit_example.csv')
dit={'Excellent':0,'Good':1,'Pass':2,'Fail':3}
train['label']=train['Quality_label'].map(dit)

feature_list=['Parameter'.format(i) for i in range(5,11)]
print(feature_list)

X_train=train[feature_list]
y_train=train['label']
X_test=test[feature_list]

oof=np.zeros((X_train.shape[0],4))    #交叉验证
prediction=np.zeros((X_test.shape[0],4))  #测试计预测

seeds=[randrange(100000)*randrange(100000) for _ in range(15)]
print('start training')
for model_seed in range(len(seeds)):
    print('模型',model_seed+1,'开始训练')
    oof_cat=np.zeros((X_train.shape[0],4))
    prediction_cat=np.zeros((X_test.shape[0],4))
    skf=StratifiedKFold(n_splits=5,random_state=seeds[model_seed],shuffle=True)
    for index,(train_index,valid_index) in enumerate(skf.split(X_train,y_train)):
        print(index)
        train_x=X_train.iloc[train_index]
        valid_x=X_train.iloc[valid_index]
        train_y=y_train.iloc[train_index]
        valid_y=y_train.iloc[valid_index]
        train_data=lgb.Dataset(train_x,label=train_y)
        validation_data=lgb.Dataset(test_X,label=test_y)
        gc.collect()
        params={
            'boosting_type':'gbdt',
            'ogjective':'multiclass',
            'num_class':4,
            'metrix':'multi_logloss',
            'learning_rate':0.025,
            'max_depth':6,
            'num_leaves':10,
            'feature_fraction':0.4
        }
        lgbmodel=lgb.train(params,train_data,valid_sets=[validation_data],num_boost_round=5000,verbose_eval=1000,
                           early_stopping_rounds=100)
        oof_cat[valid_index]+=lgbmodel.predict(valid_x)
        prediction_cat+=lgbmodel.predict(X_test)/5
        gc.collect()
    oof+=oof_cat/len(seeds)
    prediction+=prediction_cat/len(seeds)
    print('logloss',log_loss(pd.get_dummies(y_train).values,oof_cat))
print('logloss',log_loss(pd.get_dummies(y_train).values,oof))

sub=test[['Group']]
prob_cols=[i for i in submit.columns if  i  not in ['Group']]
print(prob_cols)
for i ,col in enumerate(prob_cols):
    sub[col]=prediction[:,i]
for i in prob_cols:
    sub[i]=sub.groupby('Group')[i].transform('mean')
sub=sub.drop_duplicates()
sub.to_csv("submission.csv",index=False)


