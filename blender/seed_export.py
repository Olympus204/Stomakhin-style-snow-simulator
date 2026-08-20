import bpy
import csv

from pathlib import Path
from mathutils import Vector


# Per-object settings

class SnowGroupSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Snow Group",
        default=False,
    )

    velocity: bpy.props.FloatVectorProperty(
        name="Velocity",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="VELOCITY",
    )

    mass: bpy.props.FloatProperty(
        name="Particle Mass",
        default=0.05,
        min=0.000001,
    )

    spacing: bpy.props.FloatProperty(
        name="Particle Spacing",
        default=0.05,
        min=0.0001,
    )


# Geometry helpers

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

    direction = Vector(
        (1.0, 0.371, 0.529)
    ).normalized()

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

def get_project_root():
    blend_directory = Path(
        bpy.path.abspath("//")
    )

    return blend_directory.parent

# Coordinate conversion

def blender_to_solver(vector):
    return Vector((
        vector.x,
        vector.z,
        -vector.y,
    ))
    
def get_snow_groups(context):
    objects = [
        obj
        for obj in context.scene.objects
        if (
            obj.type == "MESH"
            and obj.snow_settings.enabled
        )
    ]

    objects.sort(
        key=lambda obj: obj.name
    )

    return objects

def export_groups(
    objects,
    output_directory
):
    group_path = (
        output_directory / "groups.csv"
    )

    with group_path.open(
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file, lineterminator="\n")

        writer.writerow([
            "group_id",
            "vx",
            "vy",
            "vz",
            "mass",
        ])

        for group_id, obj in enumerate(objects):
            settings = obj.snow_settings

            blender_velocity = Vector(
                settings.velocity
            )

            solver_velocity = blender_to_solver(
                blender_velocity
            )

            writer.writerow([
                group_id,
                solver_velocity.x,
                solver_velocity.y,
                solver_velocity.z,
                settings.mass,
            ])
            
def export_particles(
    objects,
    output_directory
):
    particle_path = (
        output_directory / "particles.csv"
    )

    total_particles = 0

    with particle_path.open(
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file, lineterminator="\n")

        writer.writerow([
            "group_id",
            "x",
            "y",
            "z",
        ])

        for group_id, obj in enumerate(objects):
            settings = obj.snow_settings

            positions = seed_mesh(
                obj,
                settings.spacing
            )

            for position in positions:
                solver_position = blender_to_solver(
                    position
                )

                writer.writerow([
                    group_id,
                    solver_position.x,
                    solver_position.y,
                    solver_position.z,
                ])

            total_particles += len(positions)

            print(
                f"Group {group_id} "
                f"'{obj.name}': "
                f"{len(positions)} particles"
            )

    return total_particles

class SNOW_OT_export_groups(
    bpy.types.Operator
):
    bl_idname = "snow.export_groups"
    bl_label = "Export Snow Groups"
    bl_description = (
        "Seed and export all selected "
        "enabled snow meshes"
    )

    def execute(self, context):
        objects = get_snow_groups(
            context
        )

        if not objects:
            self.report(
                {"ERROR"},
                "No enabled snow mesh objects selected"
            )

            return {"CANCELLED"}

        project_root = get_project_root()

        output_directory = (
            project_root / "input"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        export_groups(
            objects,
            output_directory
        )

        total_particles = export_particles(
            objects,
            output_directory
        )

        self.report(
            {"INFO"},
            (
                f"Exported {len(objects)} groups "
                f"with {total_particles} particles"
            )
        )

        return {"FINISHED"}
    
class SNOW_PT_export_panel(
    bpy.types.Panel
):
    bl_label = "Snow Simulator"
    bl_idname = "SNOW_PT_export_panel"

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Snow Simulator"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj is not None and obj.type == "MESH":
            settings = obj.snow_settings

            layout.prop(
                settings,
                "enabled"
            )

            column = layout.column()
            column.enabled = settings.enabled

            column.prop(
                settings,
                "velocity"
            )

            column.prop(
                settings,
                "mass"
            )

            column.prop(
                settings,
                "spacing"
            )

        else:
            layout.label(
                text="Select a mesh to edit its settings"
            )

        layout.separator()

        groups = get_snow_groups(context)
    
        layout.label(
            text=f"Groups to Export ({len(groups)}):"
        )

        box = layout.box()

        if not groups:
            box.label(text="None")

        else:
            for group_id, group in enumerate(groups):
                box.label(
                    text=f"{group_id}: {group.name}"
                )

        layout.separator()

        if len(groups) == 1:
            button_text = "Export 1 Snow Group"
        else:
            button_text = (
                f"Export {len(groups)} Snow Groups"
            )

        layout.operator(
            "snow.export_groups",
            text=button_text
        )
        
classes = (
    SnowGroupSettings,
    SNOW_OT_export_groups,
    SNOW_PT_export_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.snow_settings = (
        bpy.props.PointerProperty(
            type=SnowGroupSettings
        )
    )


def unregister():
    del bpy.types.Object.snow_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()