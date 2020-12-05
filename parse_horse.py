from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
import bs4
import pandas as pd
import os
import glob
import re


def parse_horse_list(path):
    with open(path, 'rt', encoding='utf-8') as f:
        html = f.read()

    doc = bs4.BeautifulSoup(html, 'lxml')

    data = []
    for tr in doc.select('#contents_liquid table tr')[1:]:

        tds = tr.select('td')[1:]
        if len(tds) != 11:
            print(len(tds), path)
        href = tds[0].select_one('a')['href']
        match = re.match('/horse/(.*)/', href)
        horse_id = match[1]
        data.append([horse_id]+[e.text.strip().replace('\n', '') for e in tds])

    return data


def parse_horse_page(path):
    # 検索結果が１件だと詳細ページに強制リダイレクトされる
    with open(path, 'rt', encoding='utf-8') as f:
        html = f.read()
    doc = bs4.BeautifulSoup(html)
    match = re.match('https://db.netkeiba.com/horse/(.*)/')

    tds = doc.select_one('.db_prof_area_02 table tr td')

    return [
        match[1],
        doc.select_one('.horse_title > h1').text.strip(),
        re.match('(牝|牡)', doc.select_one('.horse_title > p.txt_01').text)[1],
        tds[0].text.strip(),
        '',
        tds[1].text.strip(),
        tds[2].text.strip(),
        tds[3].text.strip(),
        tds[6].text.strip(),

    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-s', required=True)
    args = parser.parse_args()

    horse_lists = glob.glob(os.path.join(args.src, r'https%3A%2F%2Fdb.netkeiba.com%2F%3Fpid%3Dhorse_list*'))

    pool = ProcessPoolExecutor(os.cpu_count())
    fs = [pool.submit(parse_horse_list, path) for path in horse_lists]
    concurrent.futures.wait(fs, return_when=concurrent.futures.ALL_COMPLETED)

    data = []
    for future in fs:
        data.extend(future.result())
    print(len(data))


    # horse_pages = glob.glob('https://db.netkeiba.com/horse/*')
    # print(len(horse_pages))


    columns = ['horse_id', '馬名', '性', '生年', '属性', '厩舎', '父', '母', '母父', '馬主', '生産者', '総賞金(万円)']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('horses.csv')
