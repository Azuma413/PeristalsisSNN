import numpy as np
import genesis as gs
from genesis.engine.entities.mpm_entity import MPMEntity
from genesis.engine.states.entities import MPMEntityState
from gymnasium import spaces
import gymnasium as gym

class PeristalsisEnv(gym.Env):
    def __init__(self, obs_height=1280, obs_width=720, show_viewer=False, axial_divisions=5, max_steps=1000):
        """
        腸の蠕動運動シミュレーション環境の初期化

        Args:
            obs_height (int): 観測画像の高さ
            obs_width (int): 観測画像の幅
            show_viewer (bool): ビューワーを表示するかどうか
            axial_divisions (int): 軸方向の分割数
            max_steps (int): エピソードの最大ステップ数
        """
        super().__init__()
        self.show_viewer = show_viewer
        self.obs_height = obs_height
        self.obs_width = obs_width
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

    def _setup_scene(self):
        """シミュレーションのシーンを設定"""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1e-2,
                substeps=10,
                gravity=(0, 0, 0),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0, 0.8),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=5e-4,
                lower_bound=(-1.0, -1.0, -0.5),
                upper_bound=(1.0, 4.0, 1.0),
                grid_density=128,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                visualize_mpm_boundary=True,
            ),
        )

        # パイプ（腸モデル）の追加
        self.pipe = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="3d_models/cylinder_model.stl",
                pos=(0.0, -0.05, 0.0),
                scale=0.001,
                euler=(0, 90, 90),
            ),
            material=gs.materials.MPM.Muscle(
                E=5e5,
                nu=0.45,
                rho=10000.0,
                model="neohooken",
                n_groups=6*self.axial_divisions,
                sampler="pbs",
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.4, 0.4, 0.6),
                vis_mode='visual',
            ),
        )

        # 食べ物の追加
        self.food = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=5e4,
                nu=0.4,
                rho=1000.0,
                model="neohooken",
            ),
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.025,
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.8, 0.4),
                vis_mode='visual',
            ),
        )

        # カメラの追加
        self.cam = self.scene.add_camera(
            res=(self.obs_height, self.obs_width),
            pos=(0.5, 0.1, 0.0),
            lookat=(0.0, 0.1, 0.0),
            fov=40,
            GUI=self.show_viewer
        )

        # シーンのビルド
        self.scene.build(n_envs=1)

    def _setup_muscle(self):
        """腸の筋肉を設定"""
        set_intestines_muscle(self.pipe, axial_divisions=self.axial_divisions)

    def _get_observation(self):
        """現在の状態の観測を取得"""
        # パイプの応力を取得
        stress = self.pipe.solver.get_stress().cpu().numpy()
        pos = self.pipe.get_state().pos[0].cpu().numpy()
        
        # パイプの範囲を取得
        pos_min = pos.min(0)
        pos_max = pos.max(0)
        y_range = pos_max[1] - pos_min[1]
        segment_height = y_range / self.axial_divisions
        
        # 観測値を格納する配列
        obs = np.zeros(3 * self.axial_divisions, dtype=np.float32)
        
        # 各セグメントの応力を計算
        for i in range(self.axial_divisions):
            y_start = pos_min[1] + i * segment_height
            y_end = pos_min[1] + (i + 1) * segment_height
            
            # セグメント内の粒子を選択
            mask = (pos[:, 1] >= y_start) & (pos[:, 1] < y_end)
            
            if mask.any():
                # 各セグメントの応力テンソルの主応力を計算
                segment_stress = stress[mask]
                stress_magnitude = np.linalg.norm(segment_stress, axis=1)
                
                # 3つの区画（120度ずつ）の応力を計算
                for j in range(3):
                    angle_start = j * (2 * np.pi / 3)
                    angle_end = (j + 1) * (2 * np.pi / 3)
                    
                    # 角度の計算
                    angles = np.arctan2(pos[mask, 2] - pos_min[2], pos[mask, 0] - pos_min[0])
                    angles = (angles + 2 * np.pi) % (2 * np.pi)
                    
                    # 角度範囲内の粒子を選択
                    angle_mask = (angles >= angle_start) & (angles < angle_end)
                    
                    if angle_mask.any():
                        obs[i + j * self.axial_divisions] = np.mean(stress_magnitude[angle_mask])
        
        # 観測値を0-1の範囲に正規化
        max_stress = obs.max()
        if max_stress > 0:
            obs = obs / max_stress
            
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
        self.food_initial_pos = self.food.get_state().pos[0].cpu().numpy()

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
        food_current_pos = self.food.get_state().pos[0].cpu().numpy()

        # 報酬計算：Y軸方向の移動距離
        reward = food_current_pos[1] - self.food_initial_pos[1]

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

    def save_video(self, file_name:str="save", fps=30):
        """動画を保存する関数
        
        Args:
            file_name (str): 保存するファイル名（拡張子なし）
            fps (int): フレームレート
        """
        import cv2
        from datetime import datetime
        
        # フレームの収集
        frames = []
        
        # 環境をリセット
        self.reset()
        
        # シミュレーションを実行してフレームを収集
        for _ in range(self.max_steps):
            # ランダムなアクションでシミュレーション
            action = self.action_space.sample()
            self.step(action)
            
            # フレームを取得
            frame = self.render()
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            print("フレームが収集できませんでした")
            return
        
        # 動画を保存
        output_path = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        print(f"動画を保存しています: {output_path}")
        
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

    def close(self):
        """環境をクリーンアップ"""
        if self.scene is not None:
            del self.scene

    def render(self):
        """環境の現在の状態をレンダリング"""
        if self.cam is None:
            return None
        return self.cam.render()[0]



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

if __name__ == "__main__":
    # テスト用コード
    AXIAL_DIVISIONS = 5
    
    # 初期化
    gs.init(seed=0, precision="32", debug=False, backend=gs.gpu)
    
    # シーンの作成
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        mpm_options=gs.options.MPMOptions(
            dt=5e-4,
            lower_bound=(-1.0, -1.0, -0.5),
            upper_bound=(1.0, 4.0, 1.0),
            grid_density=128,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            visualize_mpm_boundary=True,
        ),
    )
    
    # パイプの作成
    pipe = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="3d_models/cylinder_model.stl",
            pos=(0.0, -0.05, 0.0),
            scale=0.001,
            euler=(0, 90, 90),
        ),
        material=gs.materials.MPM.Muscle(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            model="neohooken",
            n_groups=6*AXIAL_DIVISIONS,
            sampler="pbs",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4, 0.6),
            vis_mode='visual',
        ),
    )
    
    # 食べ物の作成
    food = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4,
            nu=0.4,
            rho=1000.0,
            model="neohooken",
        ),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=0.025,
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.4),
            vis_mode='visual',
        ),
    )
    
    # カメラの追加
    cam = scene.add_camera(
        res=(1920, 1080),
        pos=(0.5, 0.1, 0.0),
        lookat=(0.0, 0.1, 0.0),
        fov=40,
        GUI=False
    )
    
    # シーンのビルド
    scene.build(n_envs=1)
    
    # 腸の筋肉を設定
    set_intestines_muscle(pipe, axial_divisions=AXIAL_DIVISIONS)
    
    # シーンをリセット
    scene.reset()
    
    # 蠕動運動のパラメータ
    wave_speed = 0.05  # 波の速度（小さいほど遅く進む）
    wave_length = 1.5  # 波の長さ（大きいほど長い波）
    contraction_strength = 1  # 収縮の強さ
    
    frames = []
    
    # シミュレーション実行
    for i in range(1000):
        # 各筋肉グループに対する駆動信号を作成
        actu = np.zeros(6 * AXIAL_DIVISIONS)
        
        # 縦走筋と輪走筋の協調的な動作を作成
        for j in range(AXIAL_DIVISIONS):
            # 縦走筋はその位置より後ろの部分が収縮する（波が通過した後）
            # 3つの縦走筋グループで同じパターン
            for k in range(3):
                # 波の位置に応じた駆動信号を計算
                phase = wave_speed * i - j / AXIAL_DIVISIONS * wave_length
                longitudinal_signal = 0.5 + 0.5 * np.tanh((phase) * 2)  # シグモイド関数で滑らかに変化
                # グループIDに対応する駆動信号を設定
                actu[k * AXIAL_DIVISIONS + j] = contraction_strength * longitudinal_signal
            
            # 輪走筋は波の通過中に収縮する（波の位置）
            for k in range(3):
                # 波の位置に応じた駆動信号を計算
                phase = wave_speed * i - j / AXIAL_DIVISIONS * wave_length
                circular_signal = 0.5 - 0.5 * np.cos(phase * np.pi) * np.exp(-0.5 * (phase - 0.5)**2)
                # 輪走筋が収縮するタイミングは縦走筋とは異なる
                actu[3 * AXIAL_DIVISIONS + k * AXIAL_DIVISIONS + j] = contraction_strength * circular_signal
        
        pipe.set_actuation(actu)
        scene.step()
        frames.append(cam.render()[0])
        
        # 応力値の表示
        state = pipe.get_state()
        stress = pipe.solver.get_stress()
        # print(f"Stress shape: {stress.shape}")
    
    # 動画保存のコメントアウトを解除すると保存できます
    '''
    import cv2
    from datetime import datetime
    
    output_path = f"peristalsis_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    print(f"動画を保存しています: {output_path}")
    
    height, width, _ = frames[0].shape
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"動画の保存が完了しました: {output_path}")
    '''
