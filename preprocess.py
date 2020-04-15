##### tagmynews



import json

with open('D:/cache/dataset/tagmynews/News_Category_Dataset_v2.json','r') as f:
    t = json.load(f)

tweets = []
for line in open('D:/cache/dataset/tagmynews/News_Category_Dataset_v2.json', 'r'):
    tweets.append(json.loads(line))


list = []
for i in t:
    cat = i['category']
    list.append(cat)


cat_list = ['BUSINESS','ENTERTAINMENT','HEALTH','TECH','SPORT','US','WORLD']

cnt = 0
for cat in cat_list:
    if cat not in list:
        print(cat, 'error')
    cnt += list.count(cat)
print(cnt)


with open('D:/cache/dataset/tagmynews/news','r') as f:
    t = f.readlines()


title = []
for num in range(100000):
    title.append(t[num*8].strip())


label = []
for num in range(1,100000):
    label.append(t[num*8-2].strip())


label_idx = {}
for idx, label in enumerate(set):
    label_idx[label] = idx+1


label_id = []
for i in label:
    label_id.append(label_idx[i])


# {'sport': 5, 'world': 1, 'sci_tech': 3, 'us': 2, 'business': 6, 'health': 7, 'entertainment': 4}

merged = []
for idx, line in enumerate(title):
    new_line = str(label_id[idx]) + ' ' + line
    merged.append(new_line)


import random

random.shuffle(merged)

train = merged[0:int(len(merged)*0.8)]
test = merged[int(len(merged)*0.8)::]

real_train = train[0:int(len(train)*0.8)]
val = train[int(len(train)*0.8)::]



with open('D:/cache/dataset/tagmynews/tagmynews.val.all','w') as f:
    for instance in val:
        f.write(instance)
        f.write('\n')


with open('D:/cache/dataset/twitter/trec/1/twitter_conv.data','r') as f:
    t = f.readlines()


with open('/hdd/user4/bert_classification/odp_bert_data/tagmynews//tagmynews.test.all','r') as f:
    t = f.readlines()

t1 = []
for line in t:
    new = line.split(' ',maxsplit=1)
    new[0] = str(int(new[0]) - 1)
    new = (' ').join(new)
    t1.append(new)

with open('/hdd/user4/bert_classification/odp_bert_data/tagmynews/tagmynews.test.all','w') as f:
    for line in t1:
        f.write(line)