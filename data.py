import json
import pandas as pd

with open('new_data.txt', 'r', encoding='utf-8')as file:
    data = json.loads(file.read())
new_data = data.get('data', '')

df = pd.DataFrame(columns=['text', 'c', 'y'])
text, c, y = [], [], []
for d in new_data:
    text.append(d[0])
    c.append(d[1])
    y.append(d[2])
df['text'] = text
df['c'] = c
df['y'] = y

df.to_csv('new_data.csv', index=False, encoding='utf_8_sig')

print(123)
