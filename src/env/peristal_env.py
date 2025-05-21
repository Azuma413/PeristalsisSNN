import numpy as np
import genesis as gs
from genesis.engine.entities.mpm_entity import MPMEntity
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states.entities import MPMEntityState
from gymnasium import spaces
import gymnasium as gym
import cv2
from datetime import datetime

class PeristalsisEnv(gym.Env):
    def __init__(self, img_height=1280, img_width=720, show_viewer=False, axial_divisions=5, max_steps=1000):
        """
        腸の蠕動運動シミュレーション環境の初期化

        Args:
            img_height (int): 観測画像の高さ
            img_width (int): 観測画像の幅
            show_viewer (bool): ビューワーを表示するかどうか
            axial_divisions (int): 軸方向の分割数
            max_steps (int): エピソードの最大ステップ数
        """
        super().__init__()
        self.show_viewer = show_viewer
        self.img_height = img_height
        self.img_width = img_width
        self.axial_divisions = axial_divisions
        self.max_steps = max_steps
        self.current_step = 0

        # アクション空間: 6*axial_divisions個の筋肉グループの収縮率
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6 * axial_divisions,), dtype=np.float32)
        # 観測空間: 3*axial_divisions個の領域の応力平均値
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 * axial_divisions,), dtype=np.float32)
        
        # シーンと環境の初期化
        self.scene = None
        self.pipe = None
        self.food = None
        self.cam = None
        self.food_initial_pos = None
        self.muscle_group = None

    def _setup_scene(self):
        """シミュレーションのシーンを設定"""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1e-2,
                substeps=10,
                gravity=(0, 0, -9.8),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0, 0.8),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=1e-4, # 5e-4
                lower_bound=(-1.0, -1.0, -0.5),
                upper_bound=(1.0, 4.0, 1.0),
                grid_density=128, # 128
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                visualize_mpm_boundary=True,
            ),
            show_viewer=self.show_viewer,
        )
        
        # self.cover = self.scene.add_entity(
        #     morph=gs.morphs.Mesh(
        #         file="3d_models/cylinder_model.stl",
        #         pos=(0.0, -0.05, 0.0),
        #         scale=0.001*5/3,
        #         euler=(0, 90, 90),
        #         fixed=True,
        #     ),
        #     surface=gs.surfaces.Default(
        #         color=(0.4, 1.0, 0.4, 0.6),
        #         vis_mode='visual',
        #     ),
        # )

        # パイプ（腸モデル）の追加
        self.pipe:MPMEntity = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="3d_models/cylinder_model.stl",
                pos=(0.0, -0.05, 0.0),
                scale=0.001,
                euler=(0, 90, 90),
            ),
            material=gs.materials.MPM.Muscle(
                E=5e5, # 39621.0, # [Pa] https://clinmedjournals.org/articles/jor/jor-4-048-table1.html
                nu=0.45,
                rho=10000.0,
                model="neohooken", # "corotation" or "neohooken"
                n_groups=6*self.axial_divisions,
                sampler="pbs",
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.4, 0.4, 0.6),
                vis_mode='particle',
            ),
        )

        # 食べ物の追加
        self.food:RigidEntity = self.scene.add_entity(
            # material=gs.materials.MPM.Elastic(
            #     E=1e6,
            #     nu=0.45,
            #     rho=1000.0,
            #     model="neohooken",
            # ),
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.02,
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.8, 0.4),
                vis_mode='visual',
            ),
        )

        # カメラの追加
        self.cam = self.scene.add_camera(
            res=(self.img_height, self.img_width),
            pos=(0.5, 0.1, 0.0),
            lookat=(0.0, 0.1, 0.0),
            fov=40,
            GUI=False,
        )

        # シーンのビルド
        self.scene.build(n_envs=1)

    def _setup_muscle(self):
        """腸の筋肉を設定"""
        self.muscle_group, _ = set_intestines_muscle(self.pipe, axial_divisions=self.axial_divisions)

    def _get_observation(self):
        """現在の状態の観測を取得"""
        # パイプの変形勾配を取得
        F = self.pipe.get_state().F[0].cpu().numpy()
        # pos = self.pipe.get_state().pos[0].cpu().numpy()
        obs = np.zeros((3 * self.axial_divisions,), dtype=np.float32)
        for i in range(3*self.axial_divisions, 6*self.axial_divisions):
            # 各筋肉グループの粒子のインデックスを取得
            group_indices = np.where(self.muscle_group == i)[0]
            if len(group_indices) > 0:
                # 各筋肉グループを平均化
                obs[i - 3*self.axial_divisions] = np.mean(F[group_indices, 1])
        # 観測値を0-1の範囲に正規化
        max_F = obs.max()
        if max_F > 0:
            obs = obs / max_F
        return obs

    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        self.current_step = 0

        # シーンの初期化
        gs.init(seed=seed, precision="32", debug=False, backend=gs.gpu)
        self._setup_scene()
        self._setup_muscle()

        # 食べ物の初期位置を記録
        self.food_initial_pos = self.food.get_pos()[0].cpu().numpy()

        # 初期観測を取得
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """環境を1ステップ進める"""
        # アクションの適用
        self.pipe.set_actuation(action)
        self.scene.step()
        self.current_step += 1

        # 食べ物の現在位置を取得
        food_current_pos = self.food.get_pos()[0].cpu().numpy()

        # 報酬計算：Y軸方向の移動距離
        reward = food_current_pos[1] - self.food_initial_pos[1]
        reward *= 10.0

        # 観測取得
        observation = self._get_observation()

        # 終了判定
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "food_position": food_current_pos,
            "distance_moved": reward
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """環境をクリーンアップ"""
        if self.scene is not None:
            del self.scene

    def render(self):
        """環境の現在の状態をレンダリング"""
        if self.cam is None:
            return None
        return self.cam.render()[0]


def save_video(frames, file_name:str="save", fps=30):
    """動画を保存する関数
    
    Args:
        file_name (str): 保存するファイル名（拡張子なし）
        fps (int): フレームレート
    """
    # 動画を保存
    output_path = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    print(f"動画を保存しています: {output_path}")
    
    frames = np.array(frames)
    if np.issubdtype(frames.dtype, np.floating):
        frames = (frames * 255).astype(np.uint8)
    # 最初のフレームから動画の属性を取得
    height, width, _ = frames[0].shape
    
    # VideoWriterオブジェクトを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 各フレームを書き込む
    for frame in frames:
        # OpenCVはBGR形式を使用するため、RGB形式から変換
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    # VideoWriterオブジェクトを解放
    video_writer.release()
    print(f"動画の保存が完了しました: {output_path}")

def set_intestines_muscle(robot:MPMEntity, axial_divisions=10):
    """
    腸の筋肉を設定する関数
    
    Parameters:
    - robot: 筋肉を設定するロボット（腸モデル）
    - axial_divisions: 軸方向の分割数（デフォルト: 10）
    
    長さ30cm、外半径4cm、内半径3cm（壁の厚み1cm）の腸モデルに対して、
    縦走筋（半径3.5cmより外側）と輪走筋（半径3.5cmより内側）を配置する
    """
    if isinstance(robot.material, gs.materials.MPM.Muscle): # robotがMPMの筋肉である場合
        pos = robot.get_state().pos[0] # robotの位置を取得
        n_units = robot.n_particles # robotの粒子数を取得
    else:
        raise NotImplementedError

    pos:np.ndarray = pos.cpu().numpy()
    pos_max, pos_min = pos.max(0), pos.min(0) # 最大値と最小値を取得
    pos_range = pos_max - pos_min
    axis_center = (pos_max + pos_min) * 0.5 # 中心点を計算
    
    # 軸からの距離を計算（xz平面上での距離）
    length_from_axis = np.sqrt((pos[:, 0] - axis_center[0])**2 + (pos[:, 2] - axis_center[2])**2)
    # 軸からの角度を計算（xz平面上での角度）
    angle_from_axis = np.arctan2(pos[:, 2] - axis_center[2], pos[:, 0] - axis_center[0])
    # 角度の範囲を[0, 2π]に調整
    angle_from_axis = (angle_from_axis + 2 * np.pi) % (2 * np.pi)
    # ロボットを構成する粒子の数だけの配列を作成
    muscle_group = np.zeros((n_units,), dtype=int)
    # 半径3.5cmより外側を縦走筋、内側を輪走筋とする
    mask_longitudinal = length_from_axis > 0.0354  # 縦走筋のマスク
    mask_circular = ~mask_longitudinal  # 輪走筋のマスク
    
    # 縦走筋の設定（円周方向に120度ずつ3分割、軸方向にaxial_divisions分割）
    for i in range(3):  # 円周方向の3分割
        angle_start = i * (2 * np.pi / 3)
        angle_end = (i + 1) * (2 * np.pi / 3)
        mask_angle = (angle_from_axis >= angle_start) & (angle_from_axis < angle_end)
        
        for j in range(axial_divisions):  # 軸方向の分割
            y_start = pos_min[1] + (pos_range[1] * j / axial_divisions)
            y_end = pos_min[1] + (pos_range[1] * (j + 1) / axial_divisions)
            mask_y = (pos[:, 1] >= y_start) & (pos[:, 1] < y_end)
            
            # 筋肉グループID: 縦走筋は0からaxial_divisions*3-1まで
            group_id = i * axial_divisions + j
            muscle_group[mask_longitudinal & mask_angle & mask_y] = group_id
    
    # 輪走筋の設定（円周方向に120度ずつ3分割、軸方向にaxial_divisions分割）
    for i in range(3):  # 円周方向の3分割
        angle_start = i * (2 * np.pi / 3)
        angle_end = (i + 1) * (2 * np.pi / 3)
        mask_angle = (angle_from_axis >= angle_start) & (angle_from_axis < angle_end)
        
        for j in range(axial_divisions):  # 軸方向の分割
            y_start = pos_min[1] + (pos_range[1] * j / axial_divisions)
            y_end = pos_min[1] + (pos_range[1] * (j + 1) / axial_divisions)
            mask_y = (pos[:, 1] >= y_start) & (pos[:, 1] < y_end)
            
            # 筋肉グループID: 輪走筋は3*axial_divisionsから6*axial_divisions-1まで
            group_id = 3 * axial_divisions + i * axial_divisions + j
            muscle_group[mask_circular & mask_angle & mask_y] = group_id
    
    # それぞれの粒子に対して筋肉の方向を示すベクトルを作成
    muscle_direction = np.zeros((n_units, 3), dtype=float)
    
    # 筋肉の方向を設定
    # 縦走筋はY軸方向（軸方向）のベクトル
    muscle_direction[mask_longitudinal] = np.array([0, 1, 0])
    
    # 輪走筋は円周方向のベクトル（各点での接線方向）
    for i in range(n_units):
        if mask_circular[i]:
            # 中心からの方向ベクトル（xz平面上）
            radial_vector = np.array([
                pos[i, 0] - axis_center[0],
                0,
                pos[i, 2] - axis_center[2]
            ])
            # 正規化
            if np.linalg.norm(radial_vector) > 0:
                radial_vector = radial_vector / np.linalg.norm(radial_vector)
            
            # 接線方向ベクトル（radial_vectorを90度回転させたベクトル）
            tangent_vector = np.array([-radial_vector[2], 0, radial_vector[0]])
            muscle_direction[i] = tangent_vector
            
    # 筋肉グループと筋肉の方向をもとに、robotに筋肉を設定
    robot.set_muscle(
        muscle_group=muscle_group,
        muscle_direction=muscle_direction,
    )
    return muscle_group, pos

if __name__ == "__main__":
    # テスト用コード
    AXIAL_DIVISIONS = 5
    save_frames = False
    env = PeristalsisEnv(axial_divisions=AXIAL_DIVISIONS, show_viewer=True)
    env.reset()
    frames = []
    observations = []
    wave_speed = 0.5 # [hz]
    longitudinal_strength = 1.0  # 縦走筋の収縮強度
    circular_strength = 0.0 # 輪走筋の収縮強度
    wave_length = 0.5  # 波の長さ
    # シミュレーション実行
    for i in range(1000):
        # 各筋肉グループに対する駆動信号を作成
        actu = np.zeros(6 * AXIAL_DIVISIONS)
        for j in range(AXIAL_DIVISIONS):
            phase = np.sin(wave_speed / 15 * np.pi * i + j * 2*np.pi / AXIAL_DIVISIONS)  # 波の位相
            actu[0+j] = phase * longitudinal_strength
            actu[5+j] = phase * longitudinal_strength
            actu[10+j] = phase * longitudinal_strength
            actu[15+j] = phase * circular_strength
            actu[20+j] = phase * circular_strength
            actu[25+j] = phase * circular_strength
        obs, reward, terminated, truncated, info = env.step(actu)
        print(f"Step: {i}, Reward: {reward}")
        observations.append(obs)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    if save_frames:
        save_video(frames, file_name="peristalsis_simulation", fps=30)
    # observationsを画像として保存
    obs = np.array(observations)
    obs = obs.reshape(3*AXIAL_DIVISIONS, -1)
    obs = (obs - obs.min()) / (obs.max() - obs.min())  # 正規化
    obs = (obs * 255).astype(np.uint8)  # 0-255に変換
    obs = cv2.applyColorMap(obs, cv2.COLORMAP_JET)  # カラーマップを適用
    cv2.imwrite("observations.png", obs)
    env.close()