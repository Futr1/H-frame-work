from tqdm import tqdm
import joblib
import math
import torch
def read_data(filename,predict_name):
    with open(filename, "r") as f,open(predict_name,"r") as fb:
        qid_last = 0
        doc_id = 0
        query_dic = {}
        item_list = []
        f1 = f.readlines()
        f2 = fb.readlines()
        for line,predict in tqdm(zip(f1,f2)):
            line = line[:-1]
            line = line.split(' ')[:-1]
            item = {}
            feature_list = []
            predict_socre = predict.split(' ')[-1]
            for idx in range(len(line)):
                if(idx == 0):
                    item['label'] = int(line[idx])
                elif(idx == 1):
                    item['qid'] = int(line[idx].split(':')[1])
                    if(item['qid'] == qid_last):
                        doc_id += 1
                        item['doc_id'] = doc_id
                    else:
                        qid_last = item['qid']
                        doc_id = 0
                        item['doc_id'] = doc_id
                elif(idx > 1):
                    feature_list.append(float(line[idx].split(':')[1]))
            feature_list.append(float(predict_socre))
            item['pre_score'] = float(predict_socre)
            item['feature'] = feature_list
            item_list.append(item)
        for item in tqdm(item_list):
            if not item['qid'] in query_dic.keys():
                query_dic[item['qid']] = {}
                query_dic[item['qid']][item['doc_id']] = {}
                query_dic[item['qid']][item['doc_id']]['label'] = item['label']
                query_dic[item['qid']][item['doc_id']]['pre_score'] = item['pre_score']
                query_dic[item['qid']][item['doc_id']]['feature'] = torch.tensor(item['feature'],dtype=torch.float)
            else:
                query_dic[item['qid']][item['doc_id']] = {}
                query_dic[item['qid']][item['doc_id']]['label'] = item['label']
                query_dic[item['qid']][item['doc_id']]['pre_score'] = item['pre_score']
                query_dic[item['qid']][item['doc_id']]['feature'] = torch.tensor(item['feature'],dtype=torch.float)
        joblib.dump(query_dic,'../dataset/MSLR/FOLD1/processed_data/test_top40_table_without_process.pkl')

        train_dataset = []
        for query in tqdm(query_dic.keys()):
            item = {}
            item['qid'] = query
            item['doc_id_list'] = []
            id_label = []
            for key,value in query_dic[query].items():
                item['doc_id_list'].append((query,key,value['pre_score'],value['label']))
                id_label.append((key,value['label']))


            id_label.sort(key=lambda x: (x[1], -x[0]),reverse=True)

            item['list_label'] = [x[0] for x in id_label]
            item['score_label'] = [x[1] for x in id_label]
            item['doc_id_list'].sort(key=lambda x: x[2],reverse=True)
            item['doc_and_feature'] = []
            input_seq = ''
            input_doc_list = []  #
            count = 0
            for doc in item['doc_id_list']:
                input_doc_list.append(doc)
                input_seq += '[DOC_EMB]'
                count += 1
                if count == 40:
                    break

            count = 0
            label_doc_id = []
            label_seq = ''
            label_input_id = []
            for doc in item['list_label']:
                label_seq += '[DOC_EMB]'
                label_doc_id.append((query,doc))
                for i in range(len(input_doc_list)):
                    if doc == input_doc_list[i][1]:
                        label_input_id.append(i)
                        break
                count += 1
                if(count == 40):
                    break

            train_dataset.append((input_seq,input_doc_list,label_seq,label_doc_id,label_input_id,id_label))
        joblib.dump(train_dataset, '../dataset/MSLR/FOLD1/processed_data/test_top40_without_process.pkl')

if __name__ == '__main__':
    read_data('../dataset/MSLR/FOLD1/test.txt','../dataset/MSLR/FOLD1/processed_data/lambda_mart_output_java_1000_20_ndcg10/test.predict')
    # txt data of MSLR and lambda_mart retrieve results