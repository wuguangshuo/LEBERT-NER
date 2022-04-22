from tqdm import tqdm
import codecs
import json

# 统计实体分布
def load_sentences(path):
    dict={}
    for line in tqdm(codecs.open(path,'r',encoding='utf-8')):
        line=line.strip()
        if not line:
            continue
        else:
            word=line.split(' ')
            if len(word)==2:
                tmp=word[-1].split('-')[-1]
                if tmp not in dict:
                    dict[tmp]=1
                else:
                    dict[tmp] += 1
            else:
                if word[-1]=='O':
                    if word[-1] not in dict:
                        dict[word[-1]] = 1
                    else:
                        dict[word[-1]] += 1
                else:
                    char=word[-1].split('-')[-1]
                    if char not in dict:
                        dict[char] = 1
                    else:
                        dict[char] += 1
    # del dict['O']
    res_num = sum(dict.values())
    dict_num = sorted(dict.items(), key=lambda x: x[1],reverse=True)
    with open("./data/label_dict_new", "w", encoding='utf-8') as f:
        json.dump(dict, f,sort_keys=True, ensure_ascii=False)
load_sentences('./data/train_new.txt')
print('finish')

#删除个数较少的实体
# del_label=['20','31','15','30','41','50','34','52','48','19','43','28','32','44','46','17','25','23','33','51','42','24','53','35','26']
# print(len(del_label))
# def del_sentences(path):
#     for line in tqdm(codecs.open(path,'r',encoding='utf-8')):
#         line=line.strip()
#         if not line:
#             continue
#         else:
#             word=line.split(' ')
#             if len(word)==2:
#                 tmp=word[-1].split('-')[-1]
#                 if tmp in del_label:
#                     line[-1]='O'
#             else:
#                 if word[-1]=='O':
#                     continue
#                 else:
#                     char=word[-1].split('-')[-1]
#                     if char in del_label:
#                         dict[char] = 1
#                     else:
#                         dict[char] += 1
#     res_num = sum(dict.values())
#     dict_num = sorted(dict.items(), key=lambda x: x[1],reverse=True)
#     with open("./data/label_dict_num", "w", encoding='utf-8') as f:
#         json.dump(dict_num, f,sort_keys=True, ensure_ascii=False)
# load_sentences('./data/train.txt')
# print('finish')
