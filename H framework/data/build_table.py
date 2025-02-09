import joblib
from tqdm import tqdm
train_table = joblib.load('../dataset/MSLR/FOLD1/processed_data/train_top40_table_without_process.pkl')
test_table = joblib.load('../dataset/MSLR/FOLD1/processed_data/test_top40_table_without_process.pkl')


for k,v in tqdm(test_table.items()):
    if k in train_table.keys():
        print(k)
        break
    train_table[k] = v
joblib.dump(train_table,'../dataset/MSLR/FOLD1/processed_data/feature_top40_table_without_process.pkl')
