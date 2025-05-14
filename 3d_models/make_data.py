import cadquery as cq

# パラメータ (cm単位)
length = 300.0
outer_radius = 40.0
inner_radius = 30.0
output_filename = "cylinder_model.stl"

# 3Dモデルの作成
# 外側の円筒を作成
outer_cylinder = cq.Workplane("XY").circle(outer_radius).extrude(length)

# 内側の円筒（穴）を作成
inner_cylinder = cq.Workplane("XY").circle(inner_radius).extrude(length)

# 外側の円筒から内側の円筒を引いて中空の円筒を作成
hollow_cylinder = outer_cylinder.cut(inner_cylinder)

# STLファイルとしてエクスポート
cq.exporters.export(hollow_cylinder, output_filename)

print(f"3Dモデルを {output_filename} として保存しました。")
