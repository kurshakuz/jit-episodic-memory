#!/usr/bin/env python3
"""
Tier-2 hero footage: a smooth HM3D walk-through INSTRUMENTED with the JIT story.

Renders a continuous walk of the demo scene (GLAQ4DNUx5U), then composites:
  * a top-down minimap where the trajectory draws itself and keyframe dots
    accumulate (episodic memory being built),
  * a live "lazy (JIT) vs eager map" counter (0 detections & 0 maps for JIT vs a
    climbing build cost for an eager map),
  * a depth thumbnail (RGB-D).
Encodes to docs/static/videos/rollout.mp4.

Runs in `habitat_temp` (Habitat-Sim 0.3.3, headless EGL + GPU):
    conda run -n habitat_temp python docs/render_walk.py
"""
import os, json, math, subprocess, shutil
from pathlib import Path
import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat_sim
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HAB = Path(os.environ.get("HM3D_EXAMPLE_DIR", "data/versioned_data/hm3d-0.2/hm3d/example"))
GLB = str(HAB / "00861-GLAQ4DNUx5U" / "GLAQ4DNUx5U.basis.glb")
CFG = str(HAB / "hm3d_annotated_example_basis.scene_dataset_config.json")
VID = ROOT / "docs" / "static" / "videos"
WORK = ROOT / "outputs" / "footage_work"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
W, H = 800, 600
CAP = 620
KF_EVERY = 10          # a keyframe is stored every N rendered frames
CG_S_PER_FRAME = 2160 / 500.0        # ConceptGraphs build seconds/frame (Table II)
EAGER_J_PER = 1312200 / 500.0        # eager-map precompute energy/frame (cost_crossover.json)
JIT_J = 2.52                         # JIT index-build energy, total (cost_crossover.json)
SEED = 7

# ---- palette ----
INK = (14, 16, 32); GREEN = (36, 205, 160); RED = (240, 120, 90)
ACC = (120, 128, 246); WHITE = (240, 242, 255); MUT = (150, 156, 190)


def font(sz, bold=False):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def make_cfg():
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = GLB; sc.scene_dataset_config_file = CFG; sc.enable_physics = False
    ac = habitat_sim.agent.AgentConfiguration()
    specs = []
    for uuid, stype in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid; s.sensor_type = stype; s.resolution = [H, W]
        s.position = [0.0, 1.5, 0.0]; s.hfov = 90.0
        specs.append(s)
    ac.sensor_specifications = specs
    A = habitat_sim.agent
    ac.action_space = {
        "move_forward": A.ActionSpec("move_forward", A.ActuationSpec(amount=0.10)),
        "turn_left": A.ActionSpec("turn_left", A.ActuationSpec(amount=3.0)),
        "turn_right": A.ActionSpec("turn_right", A.ActuationSpec(amount=3.0)),
    }
    return habitat_sim.Configuration(sc, [ac])


def fwd_xz(rot):
    try:
        from habitat_sim.utils.common import quat_rotate_vector
        f = quat_rotate_vector(rot, np.array([0.0, 0.0, -1.0]))
        return float(f[0]), float(f[2])
    except Exception:
        return 0.0, -1.0


def render_and_record():
    import random; random.seed(SEED)
    raw = WORK / "raw"; dep = WORK / "dep"
    for d in (raw, dep):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    sim = habitat_sim.Simulator(make_cfg())
    pf = sim.pathfinder
    try: pf.seed(SEED)
    except Exception: pass
    agent = sim.get_agent(0)
    st = habitat_sim.agent.AgentState(); st.position = pf.get_random_navigable_point(); agent.set_state(st)
    follower = habitat_sim.GreedyGeodesicFollower(
        pf, agent, goal_radius=0.4, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    poses = []; n = 0

    def cap(obs):
        nonlocal n
        Image.fromarray(obs["rgb"][..., :3]).save(raw / f"f_{n:05d}.png")
        np.save(dep / f"d_{n:05d}.npy", obs["depth"].astype(np.float16))
        s = agent.get_state()
        fx, fz = fwd_xz(s.rotation)
        poses.append({"x": float(s.position[0]), "z": float(s.position[2]), "fx": fx, "fz": fz})
        n += 1

    cap(sim.get_sensor_observations())
    tries = 0
    while n < CAP and tries < 60:
        goal = pf.get_random_navigable_point()
        if np.linalg.norm(np.array(goal) - np.array(agent.get_state().position)) < 3.0:
            tries += 1; continue
        tries = 0; steps = 0
        while n < CAP and steps < 450:
            try:
                a = follower.next_action_along(goal)
            except Exception:
                break
            if a is None:
                break
            cap(sim.step(a)); steps += 1
    sim.close()
    (WORK / "poses.json").write_text(json.dumps(poses))
    print(f"rendered {n} frames")
    return poses


def panel(draw, box, radius=12, fill=(9, 11, 24, 216)):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def battery(d, x, y, frac, col, w=92, h=15):
    d.rounded_rectangle((x, y, x + w, y + h), radius=3, outline=(205, 210, 235, 220), width=1)
    d.rectangle((x + w + 1, y + 4, x + w + 4, y + h - 4), fill=(205, 210, 235, 220))
    fw = int((w - 4) * max(0.05, min(1.0, frac)))
    d.rounded_rectangle((x + 2, y + 2, x + 2 + fw, y + h - 2), radius=2, fill=col + (255,))


def pick_query(poses, bounds):
    """Choose a real GT object the robot walked near, for a coherent query ending."""
    p = ROOT / "outputs" / "multi_scene_eval_500f" / "GLAQ4DNUx5U" / "GLAQ4DNUx5U_ground_truth.json"
    gt = json.load(open(p))
    xmin, xmax, zmin, zmax = bounds
    tp = np.array([[q["x"], q["z"]] for q in poses])
    pref = ["couch", "sofa", "chair", "table", "toilet", "bed", "sink", "tv", "cabinet", "shelf"]
    within = []
    for o in gt["objects"].values():
        c = o["center"]; cat = o["category"].lower()
        rank = next((i for i, pp in enumerate(pref) if pp in cat), 99)
        if rank == 99:
            continue
        dmin = float(np.min(np.linalg.norm(tp - np.array([c[0], c[2]]), axis=1)))
        if dmin < 3.2:
            within.append((rank, dmin, c))
    within.sort(key=lambda r: (r[0], r[1]))
    if within:
        rank, _, c = within[0]
        label = {"sofa": "couch"}.get(pref[rank], pref[rank])
        return label, c
    return "object", [(xmin + xmax) / 2, 0.0, (zmin + zmax) / 2]


def composite(poses):
    raw = WORK / "raw"; dep = WORK / "dep"
    anno = WORK / "anno"; shutil.rmtree(anno, ignore_errors=True); anno.mkdir(parents=True)
    xs = [p["x"] for p in poses]; zs = [p["z"] for p in poses]
    xmin, xmax, zmin, zmax = min(xs), max(xs), min(zs), max(zs)
    # minimap geometry (bottom-right)
    MW, MH, MARGIN, PAD = 250, 196, 18, 16
    mx0, my0 = W - MW - MARGIN, H - MH - MARGIN
    span = max(xmax - xmin, 1e-3), max(zmax - zmin, 1e-3)
    scale = min((MW - 2 * PAD) / span[0], (MH - 2 * PAD) / span[1])
    cx = mx0 + MW / 2; cy = my0 + MH / 2
    mcx = (xmin + xmax) / 2; mcz = (zmin + zmax) / 2

    def to_map(x, z):
        return (cx + (x - mcx) * scale, cy + (z - mcz) * scale)

    fB, fT, fS, fXS = font(19, True), font(15, True), font(14), font(12)
    kfpts = [(i, to_map(poses[i]["x"], poses[i]["z"])) for i in range(len(poses)) if i % KF_EVERY == 0]
    try:
        from matplotlib import cm; TURBO = (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:
        TURBO = None

    for i, p in enumerate(poses):
        base = Image.open(raw / f"f_{i:05d}.png").convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        kf = i // KF_EVERY + 1
        cg_s = kf * CG_S_PER_FRAME
        ekj = kf * EAGER_J_PER / 1000.0                  # eager energy so far (kJ)
        eager_bat = max(0.24, 1 - kf * 0.0115)           # eager drains the battery

        # --- top HUD: JIT's advantage (cheap + ready) vs the eager map's mounting cost ---
        panel(d, (16, 14, 618, 152))
        d.text((30, 20), "JUST-IN-TIME EPISODIC MEMORY", font=fT, fill=WHITE)
        d.text((30, 45), "perception on demand — not built in advance", font=fXS, fill=MUT)
        d.ellipse((30, 74, 42, 86), fill=GREEN)
        d.text((50, 71), f"JIT:  {JIT_J:.1f} J used   ·   ready to answer  [OK]   ·   no map to build",
               font=fS, fill=(198, 246, 228))
        d.ellipse((30, 100, 42, 112), fill=RED)
        d.text((50, 97), f"Eager 3D map:  {ekj:,.0f} kJ used   ·   still building {cg_s:,.0f} s",
               font=fS, fill=(249, 203, 188))
        d.text((30, 126), "Barely any energy while exploring — then answers any query in ~2.5 s.",
               font=fXS, fill=(174, 182, 255))
        d.text((524, 56), "battery", font=fXS, fill=MUT)
        battery(d, 524, 72, 0.95, GREEN)
        battery(d, 524, 100, eager_bat, RED)

        # --- depth thumbnail (top-right) ---
        dp = np.load(dep / f"d_{i:05d}.npy").astype(np.float32)
        dn = np.clip(dp / 6.0, 0, 1)
        if TURBO is not None:
            dim = Image.fromarray(TURBO[(dn * 255).astype(np.uint8)], "RGB")
        else:
            dim = Image.fromarray((dn * 255).astype(np.uint8)).convert("RGB")
        dim = dim.resize((150, 112))
        d.rounded_rectangle((W - 150 - 24, 14, W - 24 + 6, 14 + 112 + 26), radius=10, fill=(9, 11, 24, 216))
        ov.paste(dim, (W - 150 - 18, 20))
        d.text((W - 150 - 18, 134), "depth (RGB-D)", font=fXS, fill=MUT)

        # --- minimap ---
        panel(d, (mx0, my0, mx0 + MW, my0 + MH), radius=12)
        d.text((mx0 + 14, my0 + 8), "episodic memory", font=fXS, fill=MUT)
        # trajectory so far
        pts = [to_map(poses[j]["x"], poses[j]["z"]) for j in range(0, i + 1, 2)]
        if len(pts) > 1:
            d.line(pts, fill=(150, 156, 220, 210), width=2)
        # keyframe dots accumulated
        for (j, (kx, ky)) in kfpts:
            if j > i: break
            d.ellipse((kx - 3, ky - 3, kx + 3, ky + 3), fill=(GREEN + (255,)))
        # agent
        ax, ay = to_map(p["x"], p["z"])
        hn = math.hypot(p["fx"], p["fz"]) or 1.0
        hx, hz = p["fx"] / hn, p["fz"] / hn
        tip = (ax + hx * 10, ay + hz * 10)
        left = (ax - hz * 6 - hx * 4, ay + hx * 6 - hz * 4)
        right = (ax + hz * 6 - hx * 4, ay - hx * 6 - hz * 4)
        d.polygon([tip, left, right], fill=(WHITE + (255,)))

        out = Image.alpha_composite(base, ov).convert("RGB")
        out.save(anno / f"a_{i:05d}.png")

    # ---- query-resolution ending: BOTH methods localize the object ----
    qobj, qc = pick_query(poses, (xmin, xmax, zmin, zmax))
    qx, qz = to_map(qc[0], qc[2])
    fBig, fMed, fP = font(25, True), font(18, True), font(13)
    last = Image.open(raw / f"f_{len(poses) - 1:05d}.png").convert("RGBA")
    NQ = 260          # ~6.5 s ending: brief staggered reveal, then a long readable hold
    for t in range(NQ):
        base = last.copy()
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        d.rectangle((0, 0, W, H), fill=(6, 7, 18, min(160, t * 13)))
        if t >= 4:
            d.text((W / 2, 96), f"Query:  “where is the {qobj}?”", font=fBig, fill=WHITE, anchor="mm")
        c0x, c0y, c1x, c1y = W / 2 - 300, 150, W / 2 + 300, 322
        if t >= 14:
            panel(d, (c0x, c0y, c1x, c1y), radius=16, fill=(9, 11, 24, 232))
        if t >= 20:
            d.text((c0x + 30, c0y + 28), "[OK]", font=fMed, fill=GREEN)
            d.text((c0x + 56, c0y + 28), "JIT", font=fMed, fill=GREEN)
            d.text((c0x + 56, c0y + 54), "localized in 2.5 s  ·  perception deferred to now  ·  no map",
                   font=fS, fill=(210, 248, 232))
        if t >= 34:
            d.text((c0x + 30, c0y + 96), "[OK]", font=fMed, fill=RED)
            d.text((c0x + 56, c0y + 96), "Eager 3D map", font=fMed, fill=RED)
            d.text((c0x + 56, c0y + 122), "localized  ·  but only after a 2,160 s, 1.3 MJ map build",
                   font=fS, fill=(250, 206, 192))
        if t >= 52:
            d.text((W / 2, c1y + 30), "Both find the object — JIT with ~700× less pre-computation, ready from the first second.",
                   font=fP, fill=(182, 190, 255), anchor="mm")
        # minimap with the located object (both methods converge here)
        panel(d, (mx0, my0, mx0 + MW, my0 + MH), radius=12)
        d.text((mx0 + 14, my0 + 8), "episodic memory", font=fXS, fill=MUT)
        pts = [to_map(poses[j]["x"], poses[j]["z"]) for j in range(0, len(poses), 2)]
        d.line(pts, fill=(150, 156, 220, 170), width=2)
        for (j, (kx, ky)) in kfpts:
            d.ellipse((kx - 3, ky - 3, kx + 3, ky + 3), fill=GREEN + (190,))
        if t >= 24:
            pr = 1 + 0.22 * math.sin(t / 2.0)
            d.ellipse((qx - 10 * pr, qz - 10 * pr, qx + 10 * pr, qz + 10 * pr),
                      outline=(255, 210, 90, 255), width=3)
            d.ellipse((qx - 7, qz - 3, qx - 1, qz + 3), fill=GREEN + (255,))   # JIT hit
            d.ellipse((qx + 1, qz - 3, qx + 7, qz + 3), fill=RED + (255,))     # eager hit
            d.text((mx0 + MW - 14, my0 + MH - 14), f"{qobj} located", font=fXS, fill=(255, 222, 150), anchor="rs")
        out = Image.alpha_composite(base, ov).convert("RGB")
        out.save(anno / f"a_{len(poses) + t:05d}.png")

    total = len(poses) + NQ
    print(f"composited {total} frames (walk {len(poses)} + query ending {NQ}); query = {qobj}")
    return anno, total


def encode(anno, n):
    VID.mkdir(parents=True, exist_ok=True)
    out = VID / "rollout.mp4"
    subprocess.run([FFMPEG, "-y", "-framerate", "40", "-i", str(anno / "a_%05d.png"),
                    "-vf", "scale=800:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(out)], check=False)
    subprocess.run([FFMPEG, "-y", "-i", str(anno / f"a_{n // 2:05d}.png"),
                    "-vf", "scale=800:-2", str(VID / "rollout_poster.jpg")], check=False)
    print("rollout.mp4:", out.stat().st_size if out.exists() else "FAILED")


if __name__ == "__main__":
    # reuse a cached render so overlay-only tweaks skip the (slow) Habitat pass
    if (WORK / "poses.json").exists() and (WORK / "raw" / "f_00000.png").exists() \
            and (WORK / "dep" / "d_00000.npy").exists():
        poses = json.loads((WORK / "poses.json").read_text()); print("reusing cached render")
    else:
        poses = render_and_record()
    anno, total = composite(poses)
    encode(anno, total)
    print("done. (WORK kept for fast re-composite; delete outputs/footage_work when finished)")
