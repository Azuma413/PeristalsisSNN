import numpy as np
import genesis as gs

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
    morph=gs.morphs.Mesh( # 形状をobjファイルから読み込む
        file="meshes/worm/worm.obj",
        pos=(0.3, 0.3, 0.001),
        scale=0.1,
        euler=(90, 0, 0),
    ),
    material=gs.materials.MPM.Muscle(
        E=5e5, # ヤング率
        nu=0.45, # ポアソン比
        rho=10000.0, # 密度
        model="neohooken", # 応力モデル [corotation, neohooken]
        n_groups=4, # 筋肉のグループ数 ロボット内に4つの筋肉が存在できることを示す。
    ),
    surface=gs.surfaces.Default( # テクスチャの設定
        diffuse_texture=gs.textures.ImageTexture(
            image_path="meshes/worm/bdy_Base_Color.png",
        ),
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
        color    = (1.0, 0.4, 0.4),  # 色
        vis_mode = 'visual',         # 視覚モード
    ),
)

########################## build ##########################
scene.build(n_envs=1)

########################## set muscle ##########################
def set_muscle_by_pos(robot):
    if isinstance(robot.material, gs.materials.MPM.Muscle): # robotがMPMの筋肉である場合
        pos = robot.get_state().pos[0] # robotの位置を取得
        n_units = robot.n_particles # robotの粒子数を取得
    elif isinstance(robot.material, gs.materials.FEM.Muscle):
        pos = robot.get_state().pos[0, robot.get_el2v()].mean(1)
        n_units = robot.n_elements
    else:
        raise NotImplementedError

    pos:np.ndarray = pos.cpu().numpy()
    pos_max, pos_min = pos.max(0), pos.min(0) # 最大値と最小値を取得
    pos_range = pos_max - pos_min

    lu_thresh, fh_thresh = 0.3, 0.6
    muscle_group = np.zeros((n_units,), dtype=int) # ロボットを構成する粒子の数だけの配列を作成
    mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thresh) # 上下のマスクを作成
    mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thresh) # 前後のマスクを作成
    # それぞれの粒子に対して筋肉グループのマスクを作成
    muscle_group[mask_upper & mask_fore] = 0  # upper fore body
    muscle_group[mask_upper & ~mask_fore] = 1  # upper hind body
    muscle_group[~mask_upper & mask_fore] = 2  # lower fore body
    muscle_group[~mask_upper & ~mask_fore] = 3  # lower hind body
    # それぞれの粒子に対して筋肉の方向を示すベクトルを作成
    muscle_direction = np.array([[0, 1, 0]] * n_units, dtype=float)
    # 筋肉グループと筋肉の方向をもとに、robotに筋肉を設定
    robot.set_muscle(
        muscle_group=muscle_group,
        muscle_direction=muscle_direction,
    )

def set_intestines_muscle(robot, axial_divisions=10):
    # 腸の筋肉を設定する関数
    if isinstance(robot.material, gs.materials.MPM.Muscle): # robotがMPMの筋肉である場合
        pos = robot.get_state().pos[0] # robotの位置を取得
        n_units = robot.n_particles # robotの粒子数を取得
    elif isinstance(robot.material, gs.materials.FEM.Muscle):
        pos = robot.get_state().pos[0, robot.get_el2v()].mean(1)
        n_units = robot.n_elements
    else:
        raise NotImplementedError

    pos:np.ndarray = pos.cpu().numpy()
    pos_max, pos_min = pos.max(0), pos.min(0) # 最大値と最小値を取得
    pos_range = pos_max - pos_min
    # 長さ1m
    # 外半径4cm
    # 内半径3cm
    # 壁の厚み1cm
    # 軸のxz平面上の位置
    axis_coord = [pos_min[0] + pos_range[0] * 0.5, pos_min[1] + pos_range[1] * 0.5]
    # 軸からの距離を計算
    length_from_axis = np.sqrt((pos[:, 0] - axis_coord[0])**2 + (pos[:, 2] - axis_coord[1])**2)
    # 軸からの角度を計算
    angle_from_axis = np.arctan2(pos[:, 1] - axis_coord[1], pos[:, 0] - axis_coord[0])
    # 0前後, 1左右, 2上下
    muscle_group = np.zeros((n_units,), dtype=int) # ロボットを構成する粒子の数だけの配列を作成
    # 半径3.5cmより外側を縦走筋とする。
    mask_rad = length_from_axis > 0.035
    for i in range(3):
        for j in range(axial_divisions):
            # 筋肉のグループを設定 領域を上書きしていくことで正しくmaskを設定する
            muscle_group[
                mask_rad &\
                (angle_from_axis > (2*np.pi/3)*(3 - i)) &\
                (pos[:, 1] - pos_min[1] > pos_range[1]*j/axial_divisions)
            ] = i * axial_divisions + j
    # それぞれの粒子に対して筋肉の方向を示すベクトルを作成
    muscle_direction = np.zeros((n_units, 3), dtype=float)
    # 筋肉の方向を設定
    # 輪走筋はy軸周りの周方向のベクトル
    muscle_direction[~mask_rad] = np.array([
        np.cos(
    # 筋肉グループと筋肉の方向をもとに、robotに筋肉を設定
    robot.set_muscle(
        muscle_group=muscle_group,
        muscle_direction=muscle_direction,
    )

set_muscle_by_pos(pipe)

########################## run ##########################
scene.reset()
for i in range(1000):
    # 筋肉の数（n_units, n_groups）と同じ形の駆動信号配列を作成
    actu = np.array([0, 0, 0, 1.0 * (0.5 + np.sin(0.005 * np.pi * i))])

    pipe.set_actuation(actu)
    scene.step()
