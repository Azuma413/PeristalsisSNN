# PeristalsisSNN
SNNを強化学習して腸の蠕動運動を再現しようという試み。

## セットアップ
```bash
git clone --recurse-submodules https://github.com/Azuma413/PeristalsisSNN.git
cd PeristalsisSNN
uv sync
uv pip install -e "Genesis_Azuma413/.[dev]"
```
pytorchのcudaが有効化されているか確認
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
Trueと出力されればOK

## TODO
- [ ] peristal envをgymでwrapする。（[ここらへん](https://github.com/Azuma413/sound_dp/blob/main/env/genesis_env.py)を参照）

## memo
腸・ニューロンモデルの作成

腸の蠕動運動は細胞壁付近の小規模なニューラルネットワークによって制御されている
食物を特定方向に前進させるほど高い報酬を与えるようにして、R-STDPでこの運動を学習できるだろうか。
いきなり脳を再現するのではなく、体内の局所的な神経ネットワークをボトムアップ的に再現しようという試み。
key word: Peristalsis neural network

* Neuronal Control of Esophageal Peristalsis and Its Role in Esophageal Disease

サーベイ論文。 縦走筋と環状筋の収縮が同期することで、効率的に食物の輸送が行われる。 アセチルコリンは興奮性神経伝達物質として筋収縮を誘発し、一酸化窒素は抑制性神経伝達物質として筋弛緩を引き起こす。 
質点とばねで円筒状のものを作る。食物も質点とばねの集合で表現する。質点ごとに圧力をフィードバックして、SNNに入力する。e-propで食物が前方に移動するほど高い報酬を与える。SNNが円筒の周方向のばねの自然長を制御する。

genesisのソフトロボティクス機能を応用して作れないだろうか。
腸管神経系（ENS: Enteric Nervous System）
アウエルバッハ神経叢
感覚、運動、介在ニューロンの3種類がある。

* 感覚ニューロン：消化管内の物理的、化学的刺激を感知。今回は物理的な刺激に絞るとして、腸壁への圧力を入力とすれば良さそう。
* 運動ニューロン：消化管の平滑筋を収縮・弛緩させる。興奮性と抑制性運動ニューロンがある。
* 介在ニューロン：感覚ニューロンと運動ニューロンを繋ぎ、蠕動を制御。

DANやMDC-SANが有名アルゴリズム？