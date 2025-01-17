import json

def process_jsonl(file_path, file_path2):
    with open(file_path, 'r', encoding='utf-8') as file, open(file_path2, 'w', encoding='utf-8') as file2:
        for line in file:
            data = json.loads(line)

            # 截断文本至前50个单词
            words = data[2].split()
            truncated_text = ' '.join(words[:50])

            # 创建新的JSON对象
            new_data = {
                "reviewerID": data[0],
                "asin": data[1],
                "reviewText": truncated_text,
                "overall": data[3]
            }

            # 将新的JSON对象写入到文件
            file2.write(json.dumps(new_data) + '\n')

# 请根据实际文件路径进行修改
file_path = "/root/autodl-tmp/data/music/userReview.jsonl"
file_path2 = "/root/autodl-tmp/catn/movie2music/music.jsonl"
process_jsonl(file_path, file_path2)
file_path2 = "/root/autodl-tmp/catn/book2music/music.jsonl"
process_jsonl(file_path, file_path2)

file_path = "/root/autodl-tmp/data/movie/userReview.jsonl"
file_path2 = "/root/autodl-tmp/catn/movie2music/movie.jsonl"
process_jsonl(file_path, file_path2)
file_path2 = "/root/autodl-tmp/catn/book2movie/movie.jsonl"
process_jsonl(file_path, file_path2)

file_path = "/root/autodl-tmp/data/book/userReview.jsonl"
file_path2 = "/root/autodl-tmp/catn/book2music/book.jsonl"
process_jsonl(file_path, file_path2)
file_path2 = "/root/autodl-tmp/catn/book2movie/book.jsonl"
process_jsonl(file_path, file_path2)
