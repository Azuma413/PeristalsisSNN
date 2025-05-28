# uv run -m src.train_on_stable_baselines3

import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from datetime import datetime
import torch
from src.env.peristal_env import PeristalsisEnv

def train(model: str = "sac"):
    """
    PeristalsisEnv環境でSACまたはPPOエージェントを学習し、WandBにログを記録する関数
    model: "sac" または "ppo" を指定
    """
    # --- 設定 ---
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 2000000, # 総学習ステップ数
        "env_id": "PeristalsisEnv-v0",
        "axial_divisions": 5, # 環境の軸方向の分割数
        "img_height": 240, # 環境の観測画像の高さ (動画記録用)
        "img_width": 320,  # 環境の観測画像の幅 (動画記録用)
        "sac_learning_rate": 3e-4, # 学習率
        "sac_buffer_size": 500_000, # メモリ使用量削減のため縮小
        "sac_batch_size": 128,    # メモリ使用量削減のため縮小
        "sac_gamma": 0.99,
        "sac_tau": 0.005,
        "sac_train_freq": 1,
        "sac_gradient_steps": 1,
        "max_steps": 600, # 環境の最大ステップ数
    }

    run_name = f"{model}_peristalsis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"logs/{run_name}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"models/{run_name}", exist_ok=True)

    # --- WandBの初期化 ---
    run = wandb.init(
        project="PeristalRL",
        config=config,
        sync_tensorboard=True,  # stable-baselines3のTensorBoardログを同期
        monitor_gym=True,       # Gymnasium環境の情報を自動的に記録
        save_code=True,         # 学習コードを保存
        name=run_name,
    )

    # --- 環境の準備 ---
    def make_env():
        env = PeristalsisEnv(
            axial_divisions=config["axial_divisions"],
            img_height=config["img_height"],
            img_width=config["img_width"],
            show_viewer=False, # 学習中はビューワーを非表示
            max_steps=config["max_steps"],
        )
        env = Monitor(env, log_dir) # Monitorでラップ
        return env

    env = DummyVecEnv([make_env])

    # --- モデルの定義 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "sac":
        agent = SAC(
            config["policy_type"],
            env,
            learning_rate=config["sac_learning_rate"],
            buffer_size=config["sac_buffer_size"],
            batch_size=config["sac_batch_size"],
            gamma=config["sac_gamma"],
            tau=config["sac_tau"],
            train_freq=config["sac_train_freq"],
            gradient_steps=config["sac_gradient_steps"],
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )
    elif model == "ppo":
        agent = PPO(
            config["policy_type"],
            env,
            learning_rate=config["sac_learning_rate"],  # SACと同じ学習率を流用
            batch_size=config["sac_batch_size"],
            gamma=config["sac_gamma"],
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )
    else:
        raise ValueError(f"未対応のmodel指定: {model}")

    # WandbCallback: 学習のメトリクス、ハイパーパラメータ、動画などをWandBにロギング
    wandb_callback = WandbCallback(
        gradient_save_freq=100_000, # 勾配の保存頻度
        model_save_path=f"models/{run_name}/wandb_model", # WandB Artifactsとしてのモデル保存パス
        model_save_freq=50_000, # モデルの保存頻度
        log="all", # "gradients", "parameters", "env", "video" なども指定可能
        verbose=2,
    )

    callback_list = CallbackList([wandb_callback])

    # --- 学習の実行 ---
    print(f"学習を開始します。ログは {log_dir} に保存されます。")
    print(f"WandB Run: {run.url}")
    try:
        agent.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback_list,
            log_interval=1,
        )
    except Exception as e:
        print(f"学習中にエラーが発生しました: {e}")
    finally:
        # --- 学習済みモデルの保存 ---
        final_model_path = f"models/{run_name}/final_model.zip"
        agent.save(final_model_path)
        print(f"最終モデルを {final_model_path} に保存しました。")

        # --- WandBの終了 ---
        run.finish()
        print("WandBセッションを終了しました。")
        env.close()

if __name__ == "__main__":
    # 例: "sac" または "ppo" を指定
    train("ppo")
