独自に開発したgym環境を格納するためのディレクトリ。

# メモ
材料 gs.materials.MPM.Muscle を指定する際に、追加の引数 n_groups = 4 を指定します。これは、このロボットに最大4つの異なる筋肉が存在できることを意味します。

筋肉の設定には robot.set_muscle を使用します。この関数は muscle_group と muscle_direction を入力として受け取ります。どちらも長さが n_units に一致し、MPMにおける n_units は粒子数を、FEMにおける n_units は要素数を表します。

muscle_group は整数の配列（例: 0 から n_groups - 1）で、ロボットのボディのユニットが属する筋肉グループを示します。

muscle_direction は筋肉方向を指定したベクトルの浮動小数点数配列です。

このミミズの例では、ボディを4つの部分（上部前方、上部後方、下部前方、下部後方）に分割し、lu_thresh と fh_thresh を使って閾値を設定しました。

4つの筋肉グループが設定された後、set_actuation を通じて制御信号を設定する際は、入力信号は形状 (4,) の配列となります。