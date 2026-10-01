"""smoke_01_attributes.py -- adapter de-risk script #1 (run on YOUR machine).

    blender --background --factory-startup --python smoke_01_attributes.py

Validates, on your actual Blender 4.5 install, the three assumptions the
Phase-7 adapter is built on:
  A. per-vertex FLOAT point attributes: create, bulk-write (foreach_set),
     bulk-read (foreach_get), round-trip exactly;
  B. depsgraph evaluation: evaluated_get(...).to_mesh() reflects modifier
     results (vertex counts change under subsurf) -- the path that will
     capture cloth-sim geometry;
  C. triangulation access: calc_loop_triangles() yields a clean (m, 3) int
     array -- the core only ever sees triangles.

It touches NO thermal code. Prints PASS/FAIL lines; exits nonzero on failure.
Paste the full output back to me.
"""

import sys

import numpy as np

try:
    import bpy
except ImportError:
    print("FATAL: run inside Blender: blender --background --python <this file>")
    sys.exit(2)

FAILURES = []


def check(tag, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {tag}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(tag)


def main():
    print(f"[INFO] Blender {bpy.app.version_string}, python {sys.version.split()[0]}")
    check("blender >= 4.5", bpy.app.version >= (4, 5, 0),
          f"found {bpy.app.version}")

    # --- scene: UV sphere with a subsurf modifier (changes vertex count) ---
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    obj = bpy.context.active_object
    obj.name = "smoke_sphere"
    mod = obj.modifiers.new("subsurf", "SUBSURF")
    mod.levels = 1
    n_orig = len(obj.data.vertices)
    print(f"[INFO] original vertex count: {n_orig}")

    # --- A: attribute round-trip on the ORIGINAL mesh ---
    attr = obj.data.attributes.new(name="thermal_T", type="FLOAT",
                                   domain="POINT")
    check("attribute created on POINT domain",
          attr is not None and attr.domain == "POINT")

    coords = np.empty(n_orig * 3)
    obj.data.vertices.foreach_get("co", coords)
    z = coords.reshape(-1, 3)[:, 2].astype(np.float32)
    t_in = (21.0 + 15.0 * z).astype(np.float32)          # gradient by height
    attr.data.foreach_set("value", t_in)

    t_out = np.empty(n_orig, dtype=np.float32)
    obj.data.attributes["thermal_T"].data.foreach_get("value", t_out)
    check("foreach_set/foreach_get round-trip exact",
          np.array_equal(t_in, t_out),
          f"max diff {np.abs(t_in - t_out).max() if n_orig else 0}")

    # --- B: depsgraph evaluation reflects modifiers ---
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    me = eval_obj.to_mesh()
    n_eval = len(me.vertices)
    print(f"[INFO] evaluated vertex count: {n_eval}")
    check("evaluated mesh differs under subsurf (depsgraph path live)",
          n_eval > n_orig, f"{n_orig} -> {n_eval}")

    ev = np.empty(n_eval * 3)
    me.vertices.foreach_get("co", ev)
    check("evaluated coords bulk-readable & finite",
          np.isfinite(ev).all() and ev.size == n_eval * 3)

    # does our POINT attribute survive/interpolate through the modifier?
    # (informational: decides whether Phase 7 writes to original or evaluated)
    surviving = me.attributes.get("thermal_T")
    if surviving is not None and len(surviving.data) == n_eval:
        vals = np.empty(n_eval, dtype=np.float32)
        surviving.data.foreach_get("value", vals)
        print(f"[INFO] attribute PROPAGATES through subsurf "
              f"(range {vals.min():.2f}..{vals.max():.2f}) -> adapter can "
              f"write on the original mesh and let modifiers interpolate")
    else:
        print("[INFO] attribute does NOT propagate through modifiers here -> "
              "adapter must target evaluated geometry / disable modifiers "
              "at bake. NOT a failure; a design input.")

    # --- C: triangulation access ---
    me.calc_loop_triangles()
    m = len(me.loop_triangles)
    tris = np.empty(m * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    check("loop triangles present and index-valid",
          m > 0 and tris.min() >= 0 and tris.max() < n_eval,
          f"{m} triangles")

    eval_obj.to_mesh_clear()

    # --- persistence through save/load (attributes must survive .blend IO) ---
    path = bpy.app.tempdir + "smoke_01.blend"
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    obj2 = bpy.data.objects.get("smoke_sphere")
    ok = (obj2 is not None and "thermal_T" in obj2.data.attributes)
    if ok:
        t2 = np.empty(len(obj2.data.vertices), dtype=np.float32)
        obj2.data.attributes["thermal_T"].data.foreach_get("value", t2)
        ok = np.array_equal(t2, t_in)
    check("attribute survives .blend save/load exactly", ok, path)

    print(f"\n[SUMMARY] {'ALL PASS' if not FAILURES else f'FAILED: {FAILURES}'}")
    sys.exit(0 if not FAILURES else 1)


main()
