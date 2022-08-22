# coding: utf-8

import numpy as np

#サンプル行列を生成する汎用的な関数を用意しておこう。以下のinit_sample_matrixは指定したサイズの行列を生成する。各要素の数字の位には意味がある。
#
#1. 千の位：画像番号、または、フィルタ番号
#2. 百の位：チャネル番号
#3. 十の位：行番号
#4. 一の位：列番号
#
#なお、1スタートする(プログラミング上の配列は0スタートだが、この関数では1スタートを採用)

#サンプル行列の初期化
#init_sample_matrixは指定したサイズの行列を生成する。各要素の十の位が行番号を示し、一の位が列番号を示す。
def init_sample_matrix(filter_num=0, channel=0, height=6, width = 8):
    matrix = []

    for row in range(height):
        temp_row = []
        for col in range(width):
            elem = (row+1)*10 + col+1
            elem += channel * 100
            elem += filter_num * 1000
            temp_row.append(elem)
        matrix.append(temp_row)

    return np.array(matrix)
