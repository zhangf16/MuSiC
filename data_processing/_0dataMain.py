import _1dataFliter, _2itemFilter, _3userFliter

# name1 = 'Movies_and_TV'
# name2 = 'movie'

name1 = 'CDs_and_Vinyl'
name2 = 'music'

# name1 = 'Books'
# name2 = 'book'

original_csv = "/home/data/zhangfan/multimodal/originalData/" + name1 +".csv"
original_mata = "/home/data/zhangfan/multimodal/originalData/meta_" + name1 +".jsonl"
original_review = "/home/data/zhangfan/multimodal/originalData/" + name1 +".jsonl"
new_csv = "/home/data/zhangfan/multimodal/" + name2 +"/data.csv"
new_item = "/home/data/zhangfan/multimodal/" + name2 +"/itemDescription.jsonl"
new_review = "/home/data/zhangfan/multimodal/" + name2 +"/userReview.jsonl"

df = _1dataFliter.run(original_csv)
df = _2itemFilter.run(df, original_mata, new_item)
df = _3userFliter.run(df, original_review, new_review, new_item, new_csv)
