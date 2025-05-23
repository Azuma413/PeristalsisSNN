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
## 使い方
- 学習
```bash
uv run -m src.train_on_stable_baselines3
```
- 評価
```bash
uv run -m src.eval_on_sb3 --model_path models/sac_peristalsis_20250523_161642/wandb_model/model.zip
```

## TODO
- [x] peristal envをgymでwrapする。（[ここらへん](https://github.com/Azuma413/sound_dp/blob/main/env/genesis_env.py)を参照）
- [x] 圧力を取得できるようにする [参考](https://genesis-world.readthedocs.io/en/latest/_modules/genesis/engine/entities/mpm_entity.html#MPMEntity.get_state)
- [x] 腸モデルを空間に固定する（ベストは軸中心に固定、難しければ空間固定した円形の剛体などで囲む）
- [ ] 神経モデル（spikingjelly）との接続
- [x] 接触部分がうまく行っているかわからない。ルールベースで検証したい。
- [x] 塑性変形テンソルを観測として用いるとして、それをそのまま使うのか、前回の値との差を用いるのか。
- [ ] 通常のNNを用いた検証

## memo
### gs.MPMEntityState
応力より変形勾配のほうが、直接的に細胞への機械刺激を表現している。

DANやMDC-SANが有名アルゴリズム？