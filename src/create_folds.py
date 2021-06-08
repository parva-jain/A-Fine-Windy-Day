# create_folds.py
# import pandas and model_selection module of scikit-learn

import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":

    # Read training data
    df = pd.read_csv("power_gen/input/train.csv")

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    # y = df['windmill_generated_power(kW/h)'].values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, 'kfold'] = f
    
    # save the new csv with kfold column
    df.to_csv("power_gen/input/train_folds.csv", index=False)