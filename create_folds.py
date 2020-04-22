import sys
import pandas as pd
from sklearn import model_selection

if __name__== "__main__" :
    df = pd.read_csv("input/train.csv")
    print("size of df:",df.size)
    df["kfold"] = -1

    # generates samples from dataframe 

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=42)

    for fold ,(train_idx,val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(val_idx))
        # print("fold:",fold)

        df.loc[val_idx,'kfold'] = fold

    df.to_csv("input/train_folds.csv",index=False)
    df2 = pd.read_csv("input/train_folds.csv")
    print("size of df2:",df2.size)






  