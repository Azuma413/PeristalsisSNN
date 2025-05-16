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
- [x] 圧力を取得できるようにする [参考](https://genesis-world.readthedocs.io/en/latest/_modules/genesis/engine/entities/mpm_entity.html#MPMEntity.get_state)
- [ ] 腸モデルを空間に固定する（ベストは軸中心に固定、難しければ空間固定した円形の剛体などで囲む）
- [ ] 神経モデル（spikingjelly）との接続

## memo
### gs.MPMEntityState
- アフィン速度勾配
'C': <genesis.grad.tensor.Tensor>, shape: torch.Size([1, 5339, 3, 3])
- 変形勾配
'F': <genesis.grad.tensor.Tensor>, shape: torch.Size([1, 5339, 3, 3])
- 塑性ヤコビアン
'Jp': <genesis.grad.tensor.Tensor>, shape: torch.Size([1, 5339])
- 位置
'pos': <genesis.grad.tensor.Tensor>, shape: torch.Size([1, 5339, 3])
- 速度
'vel': <genesis.grad.tensor.Tensor>, shape: torch.Size([1, 5339, 3])

### 腸壁の感覚ニューロンが受け取る信号
ニューロンが受け取るのは（化学的刺激を無視すれば）機械的刺激であり、その場合、変形か応力を入力信号として用いることが考えられる。
ここでは、変形勾配よりも応力が腸壁への刺激を直接的に表現していると判断して、応力を採用する。

DANやMDC-SANが有名アルゴリズム？