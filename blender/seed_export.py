import bpy
import csv

from pathlib import Path
from mathutils import Vector


GROUP_ID = 0
VELOCITY = (0.0, -5.0, 0.0)
MASS = 1.0
SPACING = 0.25

def world_bounds(obj):
    corners = [
        obj.matrix_world @ Vector(corner)
        for corner in obj.bound_box
    ]

    min_corner = Vector((
        min(p.x for p in corners),
        min(p.y for p in corners),
        min(p.z for p in corners),
    ))

    max_corner = Vector((
        max(p.x for p in corners),
        max(p.y for p in corners),
        max(p.z for p in corners),
    ))

    return min_corner, max_corner

def point_inside_mesh(obj, world_point):
    inverse = obj.matrix_world.inverted()

    local_point = inverse @ world_point

    direction = Vector((1.0, 0.371, 0.529)).normalized()

    intersections = 0
    origin = local_point.copy()

    while True:
        hit, location, normal, face_index = obj.ray_cast(
            origin,
            direction
        )

        if not hit:
            break

        intersections += 1

        origin = location + direction * 1e-6

    return intersections % 2 == 1

def get_project_root():
    blend_directory = Path(bpy.path.abspath("//"))
    return blend_directory.parent


def get_selected_mesh():
    obj = bpy.context.active_object

    if obj is None:
        raise RuntimeError("No active object selected")

    if obj.type != "MESH":
        raise RuntimeError(
            f"Selected object '{obj.name}' is not a mesh"
        )
        
def blender_to_solver(position):
    return Vector((
        position.x,
        position.z,
        -position.y,
    ))

    return obj

def export_group(output_directory):
    group_path = output_directory / "groups.csv"

    with group_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "group_id",
            "vx",
            "vy",
            "vz",
            "mass",
        ])
        solver_VELOCITY = blender_to_solver(VELOCITY)

        writer.writerow([
            GROUP_ID,
            solver_VELOCITY[0],
            solver_VELOCITY[1],
            solver_VELOCITY[2],
            MASS,
        ])

def seed_mesh(obj, spacing):
    min_corner, max_corner = world_bounds(obj)

    positions = []

    x = min_corner.x + spacing * 0.5

    while x < max_corner.x:
        y = min_corner.y + spacing * 0.5

        while y < max_corner.y:
            z = min_corner.z + spacing * 0.5

            while z < max_corner.z:
                position = Vector((x, y, z))

                if point_inside_mesh(obj, position):
                    positions.append(position)

                z += spacing

            y += spacing

        x += spacing

    return positions

        
def export_vertices(obj, output_directory):
    particle_path = output_directory / "particles.csv"

    with particle_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "group_id",
            "x",
            "y",
            "z",
        ])

        positions = seed_mesh(obj, SPACING)

        for position in positions:
            solver_position = blender_to_solver(position)
            
            writer.writerow([
                GROUP_ID,
                solver_position.x,
                solver_position.y,
                solver_position.z,
            ])
            
            
def main():
    project_root = get_project_root()
    output_directory = project_root / "input"

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    obj = get_selected_mesh()

    export_group(output_directory)
    export_vertices(obj, output_directory)

    print(
        f"Exported seed data from '{obj.name}' "
        f"to {output_directory}"
    )


if __name__ == "__main__":
    main()