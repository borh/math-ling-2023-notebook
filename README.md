# 学術論文形成を支える接続表現と前後文末モダリティとの共起構造

(English follows)

このリポジトリでは，以下の論文で使用した分析データとコードを共有しています。

[ホドシチェク・ボル，阿辺川武，仁科喜久子，ベケシュ・アンドレイ（2023）「学術論文形成を支える接続表現と前後文末モダリティとの共起構造―談話分析の視点から―」『計量国語学』34巻1号, pp. 1-16](https://www.jstage.jst.go.jp/article/mathling/34/1/34_1/_article/-char/ja)

## 要旨

> 本稿は日本語非母語話者で卒業論文，学術論文を書く必要がある学生に有用な支援システムを構築する研究の一環である．その一部として，日本語による学術系コーパスから接続表現と文末モダリティの共起関係を考察する．はじめに1文中の接続表現とその文末モダリティの2項目共起の分布を計量的に分析し，次に接続表現の前文にあるモダリティとの関係を3項目共起として同様に分析する．その中で共起尺度の高い組み合わせに注目し，談話分析の視点から観察することで，ジャンルとしての学術論文の談話構造の特徴を示し，学習者コーパスと論文コーパスを比較する．そこにみられる学習者が補うべき談話構造上の問題点を見出し，学術論文のスタイル習得を可能にする論文作成支援を目指す．

## ノートブック

ノートブックの実行結果は下記から確認できます：

-   [分析ノートブック](stats.html)

## 再現方法
リポジトリ内の[stats.jl](stats.jl)ファイルを使用し，[Julia 1.9](https://julialang.org/downloads/)および[Pluto.jl](https://plutojl.org/)を用いて結果を再現できます。

```bash
julia --project=. -e 'import Pluto; Pluto.run()'
```

（Julia 1.10では，一部の可視化が正常に表示されない。詳しくはJuliaの[Issue 52385](https://github.com/JuliaLang/julia/issues/52385)を参照。）

## Bibtex

```bibtex
@article{hodoscek2023japanese,
  author    = {ホドシチェク・ボル and 阿辺川武 and 仁科喜久子 and ベケシュ・アンドレイ},
  title     = {学術論文形成を支える接続表現と前後文末モダリティとの共起構造―談話分析の視点から―},
  journal   = {計量国語学},
  volume    = {34},
  number    = {1},
  date      = {2023-06-20},
  pages     = {1-16},
  keywords  = {談話分析, ジャンル, 論文作成支援, アカデミック・ライティング, 学術系コーパス, BCCWJ, 自己相互情報量, エントロピー, 日本語学習者},
  note      = {特集・論文A},
  language  = {japanese},
}

```

## サイトマップ

[Githubリポジトリ](https://github.com/borh/math-ling-2023-notebook/)内のファイル一覧：

```
  4.0 KiB ┌─ LICENSE
  4.0 KiB ├─ README.md
 84.0 KiB ├─ stats.jl
  3.4 MiB ├─ stats.html
  4.0 KiB ├─ Project.toml
 12.0 KiB ├─ 人科社BDB基本統計.xlsx
  4.0 KiB ├─ 学習者DB基本統計.xlsx
 42.9 MiB ├─ ジャンル別統計-2022-07-17.xlsx
 11.0 MiB ├─ 人科社B-共起データ-2022-07-17.xlsx
480.0 KiB ├─ 学習者-共起データ-2022-07-17.xlsx
  1.4 MiB ├─ 学習者統計-2022-07-17.xlsx
 25.4 MiB ├─ B科人社統計-2022-07-17.xlsx
          math-ling-2023-notebook
```

# Cooccurrence structures of conjunctive expressions and surrounding sentence-final modality forms supporting the composition of academic papers: A discourse-analytical perspective

This repository shares the analysis data and code used in the following paper:

[Hodošček, B., Abekawa, T., Nishina, K., & Bekeš, A. (2023). Cooccurrence structures of conjunctive expressions and surrounding sentence-final modality forms supporting the composition of academic papers: A discourse-analytical perspective. _Mathematical Linguistics, 34_(1), 1–16.](https://www.jstage.jst.go.jp/article/mathling/34/1/34_1/_article/-char/en)

## Abstract

> This paper is part of a larger research project aimed at building a composition support system for L2 Japanese learners writing academic reports, graduation theses, and academic papers. As an extension to previous research on conjunctive expressions, this research examines the cooccurrence relationship between conjunctive expressions and sentence-final modality forms using a variety of written corpora including research papers, textbooks, a subset of the Balanced Corpus of Contemporary Written Japanese, and several academic learner corpora. First, we quantitatively analyze the cooccurrence distributions of conjunctive expression and sentence-final modality form pairs, and then extend the analysis to also include sentence-final modality forms from the preceding sentence. By focusing on the most salient and typical collocations using a combination of corpus-normalized frequencies, pointwise mutual information, and entropy, and observing them from the perspective of discourse analysis, we uncover the characteristics of discourse structure in the genre of academic writing and compare them with those of L2 learner academic writing. Through the process of identifying problems in the discourse structure of learner corpora compared to reference academic corpora, we aim to help learners write conjunctive expressions and sentence-final modality forms in the appropriate academic style.

## Notebook

You can view the results of the notebook execution here:

[Analysis Notebook](stats.html)

## Reproduction

You can reproduce the results using [Julia 1.9](https://julialang.org/downloads/) and [Pluto.jl](https://plutojl.org/) by running the [stats.jl](stats.jl) file in the repository.

```bash
julia --project=. -e 'import Pluto; Pluto.run()'
```

This will open up a browser window from which you can execute and inspect individual cells or the whole notebook.

(Note: In Julia 1.10, some visualizations do not currently render correctly. For more details, refer to [this issue](https://github.com/JuliaLang/julia/issues/52385).)

## Bibtex

```bibtex
@article{hodoscek2023english,
  author    = {HODOŠČEK, Bor and ABEKAWA, Takeshi and NISHINA, Kikuko and BEKEŠ, Andrej},
  title     = {Cooccurrence Structures of Conjunctive Expressions and Surrounding Sentence-Final Modality Forms Supporting the Composition of Academic Papers: A Discourse-Analytical Perspective},
  journal   = {Mathematical Linguistics},
  volume    = {34},
  number    = {1},
  date      = {2023-06-20},
  pages     = {1-16},
  keywords  = {discourse analysis, genre, research paper composition assistance, academic writing, academic corpus, BCCWJ, pointwise mutual information, entropy, Japanese language learners},
  note      = {Paper (A) for the 2022 Special Section},
  language  = {english},
}
```

## Sitemap

[The Github repository](https://github.com/borh/math-ling-2023-notebook/) contains the following files:

```
  4.0 KiB ┌─ LICENSE
  4.0 KiB ├─ README.md
 84.0 KiB ├─ stats.jl
  3.4 MiB ├─ stats.html
  4.0 KiB ├─ Project.toml
 12.0 KiB ├─ 人科社BDB基本統計.xlsx
  4.0 KiB ├─ 学習者DB基本統計.xlsx
 42.9 MiB ├─ ジャンル別統計-2022-07-17.xlsx
 11.0 MiB ├─ 人科社B-共起データ-2022-07-17.xlsx
480.0 KiB ├─ 学習者-共起データ-2022-07-17.xlsx
  1.4 MiB ├─ 学習者統計-2022-07-17.xlsx
 25.4 MiB ├─ B科人社統計-2022-07-17.xlsx
          math-ling-2023-notebook
```

