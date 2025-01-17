import pandas as pd
 
# 读取CSV文件
df = pd.read_csv('/root/code/DiffCDR-main/data/mid/Movies_and_TV.csv')
 
df.rename(columns={'user_id': 'uid',
                   'parent_asin': 'iid',
                   'rating': 'y'
                   }, inplace=True)
 
# 将更改后的DataFrame写回到CSV文件
df.to_csv('/root/code/DiffCDR-main/data/mid/Movies_and_TV.csv', index=False)

df = pd.read_csv('/root/code/DiffCDR-main/data/mid/CDs_and_Vinyl.csv')
 
df.rename(columns={'user_id': 'uid',
                   'parent_asin': 'iid',
                   'rating': 'y'
                   }, inplace=True)
 
# 将更改后的DataFrame写回到CSV文件
df.to_csv('/root/code/DiffCDR-main/data/mid/CDs_and_Vinyl.csv', index=False)

df = pd.read_csv('/root/code/DiffCDR-main/data/mid/Books.csv')
 
df.rename(columns={'user_id': 'uid',
                   'parent_asin': 'iid',
                   'rating': 'y'
                   }, inplace=True)
 
# 将更改后的DataFrame写回到CSV文件
df.to_csv('/root/code/DiffCDR-main/data/mid/Books.csv', index=False)