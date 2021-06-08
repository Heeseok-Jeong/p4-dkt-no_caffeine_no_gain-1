import numpy as np
import pandas as pd
import os

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

# 데이터 경로 맞춰주세요!
DATA_PATH = '/opt/ml/input/data/train_dataset'
train_org_df = pd.read_csv(os.path.join(DATA_PATH, "train_data.csv"), dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
test_org_df = pd.read_csv(os.path.join(DATA_PATH, "test_data.csv"), dtype=dtype, parse_dates=['Timestamp'])
test_org_df = test_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

def feature_engineering(df):

    # 문항이 중간에 비어있는 경우를 파악 (1,2,3,,5)
    def assessmentItemID2item(x):
        return int(x[-3:]) - 1  # 0 부터 시작하도록 
    df['item'] = df.assessmentItemID.map(assessmentItemID2item)

    item_size = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
    testId2maxlen = item_size.to_dict() # 중복해서 풀이할 놈들을 제거하기 위해

    item_max = df.groupby('testId').item.max()
    shit_index = item_max[item_max +1 != item_size].index
    shit_df = df.loc[df.testId.isin(shit_index),['assessmentItemID', 'testId']].drop_duplicates().sort_values('assessmentItemID')      
    shit_df_group = shit_df.groupby('testId')

    shitItemID2item = {}
    for key in shit_df_group.groups:
        for i, (k,_) in enumerate(shit_df_group.get_group(key).values):
            shitItemID2item[k] = i
        
    def assessmentItemID2item_order(x):
        if x in shitItemID2item:
            return int(shitItemID2item[x])
        return int(x[-3:]) - 1  # 0 부터 시작하도록 
    df['item_order'] =  df.assessmentItemID.map(assessmentItemID2item_order)



    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
    df_group = df.groupby(['userID','testId'])['answerCode']
    df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
    df['user_total_ans_cnt'] = df_group.cumcount()
    df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

    # 유저가 푼 시험지에 대해, 유저의 풀이 순서 계산 (시험지를 반복해서 풀었어도, 누적되지 않음)
    # 특정 시험지를 얼마나 반복하여 풀었는지 계산 ( 2번 풀었다면, retest == 1)
    df['test_size'] = df.testId.map(testId2maxlen)
    df['retest'] = df['user_total_ans_cnt'] // df['test_size']
    df['user_test_ans_cnt'] = df['user_total_ans_cnt'] % df['test_size']

    # 각 시험지 당 유저의 정확도를 계산
    df['user_test_correct_cnt'] = df.groupby(['userID','testId','retest'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_acc'] = df['user_test_correct_cnt']/df['user_test_ans_cnt']

    # 본 피처는 train에서 얻어진 값을 그대로 유지합니다.
    df["test_mean"] = df.testId.map(testId_mean_sum['mean'])
    df['test_sum'] = df.testId.map(testId_mean_sum['sum'])
    df["ItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
    df['ItemID_sum'] = df.assessmentItemID.map(assessmentItemID_mean_sum['sum'])
    df["tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])
    df['tag_sum'] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['sum'])

    return df

# trian에서 각 문제 평균 뽑기

testId_mean_sum = train_org_df.groupby(['testId'])['answerCode'].agg(['mean','sum']).to_dict()
assessmentItemID_mean_sum = train_org_df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
KnowledgeTag_mean_sum = train_org_df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()

# 맞춰야하는 문항 ID 파악
set_assessmentItemID = set(test_org_df.loc[test_org_df.answerCode == -1, 'assessmentItemID'].values)
test = test_org_df[test_org_df['userID']==test_org_df['userID'].shift(-1)]

train = feature_engineering(train_org_df)
test = feature_engineering(test)
# 피처를 대충만들어서 꽤 오래걸립니다.

train = train.fillna(0)

acc = train.loc[train.userID!=train.userID.shift(-1), ["userID", "user_total_acc", "user_total_ans_cnt"]].sort_values("user_total_acc")

u_id = []
for i in range(len(acc.userID.values)):
    if i % 10 in [0, 5]:
        u_id.append(acc.userID.values[i])

print(f"valid로 들어가는 user 비율: {len(u_id) / acc.shape[0]}")
print(f"valid로 들어가는 row 비율: {train_org_df.loc[train_org_df.userID.isin(u_id), :].shape[0] / train_org_df.shape[0]}")

train_org_df = pd.read_csv(os.path.join(DATA_PATH, "train_data_add_elapsed.csv"), dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

train_org_df.loc[~train_org_df.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "fixed_train.csv"), index=False)
train_org_df.loc[train_org_df.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "fixed_valid.csv"), index=False)

print(f"make fixed_data done")