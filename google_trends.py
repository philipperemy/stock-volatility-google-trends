import os

import wget

trends = open('trends_url_list.txt').readlines()

output_dir = 'trends/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Downloading the trends...')
for trend in trends:
    wget.download(trend.strip(), out=output_dir)
print('Download done.')