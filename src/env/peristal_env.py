import numpy as np
import genesis as gs

AXIAL_DIVISIONS = 5
########################## init ##########################
gs.init(seed=0, precision="32", logging_level="debug")

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-2, # simulation time step. default=1e-2
        substeps=10,
        # gravity=(0, 0, 0), 
        gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    mpm_options=gs.options.MPMOptions(
        dt=5e-4,
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        visualize_mpm_boundary=True,
    ),
)

########################## entities ##########################
scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        coup_friction=5.0,
    ),
)

pipe = scene.add_entity( # urdfのパイプを作る
    morph=gs.morphs.Mesh( # 形状をstlファイルから読み込む
        file="3d_models/cylinder_model.stl",
        pos=(0.3, 0.3, 0.001),
        scale=1.0,
        euler=(0, 0, 0),
    ),
    material=gs.materials.MPM.Muscle(
        E=5e5, # ヤング率
        nu=0.45, # ポアソン比
        rho=10000.0, # 密度
        model="neohooken", # 応力モデル [corotation, neohooken]
        n_groups=6*AXIAL_DIVISIONS, # 筋肉のグループ数
    ),
    surface=gs.surfaces.Default( # テクスチャの設定
        color    = (1.0, 0.4, 0.4),  # 色
        vis_mode = 'visual',         # 視覚モード
    ),
)

food = scene.add_entity( # 食べ物を模したMPMの球体を追加する
    material=gs.materials.MPM.Elastic(
        E=5e4, # 高いほど硬い
        nu=0.4, # 低いほど潰れやすい 0 < nu < 0.5
        rho=1000.0,
        model="neohooken",
    ),  # 弾性材料
    morph=gs.morphs.Sphere(
        pos  = (0.0, -0.5, 0.25),  # 位置
        radius=0.1, # 半径[m]
    ),
    surface=gs.surfaces.Default(
        color    = (0.8, 0.8, 0.4),  # 色
        vis_mode = 'visual',         # 視覚モード
    ),
)

########################## build ##########################
scene.build(n_envs=1)

########################## set muscle ##########################

def set_intestines_muscle(robot, axial_divisions=10):
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
    elif isinstance(robot.material, gs.materials.FEM.Muscle):
        pos = robot.get_state().pos[0, robot.get_el2v()].mean(1)
        n_units = robot.n_elements
    else:
        raise NotImplementedError

    pos: np.ndarray = pos.cpu().numpy()
    pos_max, pos_min = pos.max(0), pos.min(0) # 最大値と最小値を取得
    pos_range = pos_max - pos_min
    
    # 長さ30cm
    # 外半径4cm
    # 内半径3cm
    # 壁の厚み1cm
    
    # 軸のxy平面上の位置（中心点）を計算
    axis_center_x = pos_min[0] + pos_range[0] * 0.5
    axis_center_z = pos_min[2] + pos_range[2] * 0.5
    
    # 軸からの距離を計算（xz平面上での距離）
    length_from_axis = np.sqrt((pos[:, 0] - axis_center_x)**2 + (pos[:, 2] - axis_center_z)**2)
    
    # 軸からの角度を計算（xz平面上での角度）
    angle_from_axis = np.arctan2(pos[:, 2] - axis_center_z, pos[:, 0] - axis_center_x)
    # 角度の範囲を[0, 2π]に調整
    angle_from_axis = (angle_from_axis + 2 * np.pi) % (2 * np.pi)
    
    # ロボットを構成する粒子の数だけの配列を作成
    muscle_group = np.zeros((n_units,), dtype=int)
    
    # 半径3.5cmより外側を縦走筋、内側を輪走筋とする
    mask_longitudinal = length_from_axis > 0.035  # 縦走筋のマスク
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
                pos[i, 0] - axis_center_x,
                0,
                pos[i, 2] - axis_center_z
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

set_intestines_muscle(pipe, axial_divisions=AXIAL_DIVISIONS) # 腸の筋肉を設定

########################## run ##########################
scene.reset()
for i in range(1000):
    actu = np.array([0, 0, 0, 1.0 * (0.5 + np.sin(0.005 * np.pi * i))])
    pipe.set_actuation(actu)
    scene.step()