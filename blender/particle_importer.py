import bpy
import csv

from pathlib import Path


OBJECT_NAME = "SnowParticles"
MESH_NAME = "SnowParticlesMesh"

def get_project_root():
    blend_directory = Path(bpy.path.abspath("//"))
    return blend_directory.parent

def get_frame_path(frame_no):
    project_root = get_project_root()

    return (
        project_root
        / "output"
        / f"frame_{frame_no:06d}.csv"
    )
    
def read_positions(frame_path):
    positions = []

    if not frame_path.exists():
        raise RuntimeError(
            f"Frame file does not exist: {frame_path}"
        )

    with frame_path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            position = (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            )

            positions.append(position)

    return positions

def create_particle_object(positions):
    mesh = bpy.data.meshes.new(MESH_NAME)

    mesh.from_pydata(
        positions,
        [],
        []
    )

    mesh.update()

    obj = bpy.data.objects.new(
        OBJECT_NAME,
        mesh
    )

    bpy.context.collection.objects.link(obj)

    return obj

def update_particle_object(positions):
    obj = bpy.data.objects.get(OBJECT_NAME)

    if obj is None:
        return create_particle_object(positions)

    mesh = obj.data

    if len(mesh.vertices) != len(positions):
        mesh.clear_geometry()

        mesh.from_pydata(
            positions,
            [],
            []
        )

        mesh.update()

        return obj

    for vertex, position in zip(
        mesh.vertices,
        positions
    ):
        vertex.co = position

    mesh.update()

    return obj

def load_frame(frame_no):
    frame_path = get_frame_path(frame_no)

    positions = read_positions(frame_path)

    update_particle_object(positions)

    print(
        f"Loaded frame {frame_no}: "
        f"{len(positions)} particles"
    )
    
def snow_frame_change(scene):
    frame_no = scene.frame_current

    frame_path = get_frame_path(frame_no)

    if not frame_path.exists():
        return

    positions = read_positions(frame_path)

    update_particle_object(positions)
    
def register_frame_handler():
    for handler in list(
        bpy.app.handlers.frame_change_pre
    ):
        if getattr(
            handler,
            "__name__",
            ""
        ) == "snow_frame_change":
            bpy.app.handlers.frame_change_pre.remove(
                handler
            )

    bpy.app.handlers.frame_change_pre.append(
        snow_frame_change
    )
    
def main():
    register_frame_handler()

    frame_no = bpy.context.scene.frame_current

    load_frame(frame_no)


if __name__ == "__main__":
    main()
