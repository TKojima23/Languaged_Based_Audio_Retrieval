# Languaged_Based_Audio_Retrieval

テキストクエリから音声を検索するためのアプリです。  
音声埋め込みモデルとして [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) を利用しています。

このアプリは、主に **10秒区間ベースの Language-based Audio Retrieval** を想定しています。  
加えて、エネルギーベースの音声区間検出により、無音区間をなるべく除いた検索も行えるようにしています。
10秒以上の波形は5秒ごとの重複を許して10秒の波形として音声の埋め込みを行います。

## Features

- テキストから音声ファイルを検索
- LAION-CLAP による音声・テキスト埋め込み
- 基本は **10秒区間単位** の検索
- エネルギーベースの音声区間検出により、音がある部分のみを対象とした検索にも対応

## Repository Structure

```text
.
├─ audio_retrieval_app.py
├─ config.py
├─ checkpoints/
│  └─ 630k-audioset-best.pt
├─ CLAP/
│  └─ ...
└─ ...
```

## Requirements

このアプリを動かすには、LAION-CLAP 本体と学習済みチェックポイントを手動で配置する必要があります。

### 1. CLAP を clone する

リポジトリ直下に `CLAP` フォルダを作成し、その中に [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) を clone してください。

```bash
git clone https://github.com/LAION-AI/CLAP.git CLAP
```

### 2. チェックポイントを配置する

リポジトリ直下に `checkpoints` フォルダを作成し、その中に `630k-audioset-best.pt` を配置してください。

```text
checkpoints/630k-audioset-best.pt
```

## CLAP の修正が必要な場合

環境によっては、CLAP のチェックポイント読み込み時にエラーが発生する場合があります。(PyTorchのバージョン依存)  
その場合は以下の修正を行ってください。

対象ファイル

CLAP/src/laion_clap/clap_module/factory.py

54行目付近にある `torch.load()` の呼び出しを修正します。

修正前

```python
torch.load(...)
```

修正後

```python
torch.load(..., weights_only=False)
```

## Configuration

検索対象となる音声ファイルのディレクトリを `config.py` で指定します。

`config.py` の以下の変数を設定してください。

```python
CLOTHO_AUDIO_DIR = r"/path/to/your/audio/files"
```

## How to Run

リポジトリ直下で以下を実行するとアプリが起動します。

```bash
python audio_retrieval_app.py
```

検索は **10 秒区間ベース** で行われます。

また、エネルギーベースの音声区間検出を利用し、無音部分を除いた音声区間のみを対象として検索することもできます。  
これにより、無音が長い音声でも検索結果の精度が改善されます。

## Acknowledgements

This project uses [LAION-CLAP](https://github.com/LAION-AI/CLAP).

If you use this repository, please also refer to the original LAION-CLAP repository and its license.