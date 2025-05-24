# 学習済みSACポリシーを使って環境を動作させ、動画を保存するスクリプト
# 使い方例: uv run -m src.eval_on_sb3 --model_path models/sac_peristalsis_xxxx/best_model.zip

import argparse
import os
import numpy as np
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.peristal_env import PeristalsisEnv

def make_env(axial_divisions=5, img_height=240, img_width=320, max_steps=600):
    env = PeristalsisEnv(
        axial_divisions=axial_divisions,
        img_height=img_height,
        img_width=img_width,
        show_viewer=False,
        max_steps=max_steps,
    )
    return env

def main():
    parser = argparse.ArgumentParser(description="学習済みポリシーで環境を動作させ、動画を保存します。")
    parser.add_argument("--model_path", type=str, required=True, help="学習済みモデルのパス (.zip)")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="保存する動画ファイル名")
    parser.add_argument("--episode_length", type=int, default=600, help="1エピソードのステップ数")
    parser.add_argument("--axial_divisions", type=int, default=5, help="環境の軸方向分割数")
    parser.add_argument("--img_height", type=int, default=1280, help="画像の高さ")
    parser.add_argument("--img_width", type=int, default=720, help="画像の幅")
    args = parser.parse_args()

    # 環境の作成
    env = DummyVecEnv([lambda: make_env(
        axial_divisions=args.axial_divisions,
        img_height=args.img_height,
        img_width=args.img_width,
        max_steps=args.episode_length
    )])

    # モデルのロード
    model = SAC.load(args.model_path, env=env)

    obs = env.reset()
    frames = []

    for step in range(args.episode_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # フレーム取得
        frame = env.envs[0].render(mode="rgb_array")
        frames.append(frame)
        if done[0]:
            break

    # 動画保存（imageioを使用）
    # フレームはRGB配列なのでそのまま保存可能
    imageio.mimsave(args.output_video, frames, fps=30)
    print(f"動画を {args.output_video} に保存しました。")

    env.close()

if __name__ == "__main__":
    main()
