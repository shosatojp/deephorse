# deephorse

## `parse_horse.py`

`netkeiba.com`から取得した馬一覧をパース

## `pedigree.py`

先祖を調べる

## `blood2vec`

- CBOWで血統から馬の分散表現を獲得する
- 親等により割引率を設定する

- 騎手、調教師、馬主は担当したことのある馬ベクトルの、時間経過による割引率を考慮した和とする（？）
- 木構造を考慮したCBOWの設計
