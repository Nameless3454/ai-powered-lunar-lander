"""
PPO Lunar Lander - Cinematic Scene Renderer (portrait, LinkedIn-ready)

Your trained PPO agent flies a rocket in a stylised dusk launch-pad scene:
sunset sky, landing pad, live score billboard, engine flames, smoke,
explosions and subtitles. Physics still come from Gymnasium LunarLander-v3.

Requires: gymnasium[box2d], stable-baselines3, pygame
Optional for --record: pip install imageio imageio-ffmpeg
"""
import argparse
import math
import random
import sys

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO

CW, CH = 720, 1280          # canvas (9:16 portrait)
PAD_X, PAD_Y = 360, 960     # landing pad centre (rocket feet touch PAD_Y)
HORIZON = 740
ALT_SCALE = 450             # pixels per unit of observation altitude
X_SCALE = 300               # pixels per unit of observation x
FPS = 50

WHITE = (255, 255, 255)
AMBER = (240, 170, 60)
GREEN = (80, 222, 130)
RED = (255, 95, 95)

CAPTIONS = [
    "Coasting... the AI is calculating",
    "Left thruster correcting",
    "Main engine braking",
    "Right thruster correcting",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_baseline.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--title", default="AI ROCKET|LANDING")
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", default="demo.mp4")
    p.add_argument("--scale", type=float, default=0.6, help="window size factor")
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vgradient(surf, rect, stops):
    x, y, w, h = rect
    for i in range(h):
        t = i / max(h - 1, 1)
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0
                pygame.draw.line(surf, lerp(c0, c1, k), (x, y + i), (x + w - 1, y + i))
                break


def er(cx, cy, w, h):
    return pygame.Rect(cx - w // 2, cy - h // 2, w, h)


def spaced(font, text, color, spacing):
    parts = [font.render(c, True, color) for c in text]
    w = sum(s.get_width() for s in parts) + spacing * (len(text) - 1)
    out = pygame.Surface((max(w, 1), font.get_height()), pygame.SRCALPHA)
    x = 0
    for s in parts:
        out.blit(s, (x, 0))
        x += s.get_width() + spacing
    return out


def outlined(canvas, font, text, color, center, shadow=3):
    t = font.render(text, True, color)
    s = font.render(text, True, (0, 0, 0))
    r = t.get_rect(center=center)
    canvas.blit(s, r.move(shadow, shadow))
    canvas.blit(t, r)


# --------------------------------------------------------------- scene art
def build_background():
    rnd = random.Random(7)
    bg = pygame.Surface((CW, CH))

    # sky
    vgradient(bg, (0, 0, CW, HORIZON),
              [(0, (62, 52, 150)), (0.5, (150, 100, 165)), (1, (250, 172, 112))])
    for _ in range(45):
        pygame.draw.circle(bg, (235, 230, 255),
                           (rnd.randint(0, CW), rnd.randint(0, 320)), 1)

    # sun + glow
    glow = pygame.Surface((400, 400), pygame.SRCALPHA)
    for r in range(200, 0, -8):
        pygame.draw.circle(glow, (255, 210, 140, int(110 * (1 - r / 200)) + 6),
                           (200, 200), r)
    bg.blit(glow, (10 - 200, 650 - 200))
    pygame.draw.circle(bg, (255, 228, 160), (10, 650), 70)

    # mountains
    for col, base, amp in [((120, 92, 150), HORIZON - 40, 40),
                           ((150, 100, 125), HORIZON - 10, 22)]:
        ph = rnd.random() * 6
        pts = [(0, HORIZON)]
        for x in range(0, CW + 20, 20):
            y = base - amp * (0.5 + 0.5 * math.sin(x / 90 + ph)) \
                - amp * 0.4 * math.sin(x / 37 + ph * 2)
            pts.append((x, y))
        pts.append((CW, HORIZON))
        pygame.draw.polygon(bg, col, pts)

    # control building (right)
    pygame.draw.rect(bg, (150, 118, 108), (480, 560, 260, HORIZON - 560))
    pygame.draw.rect(bg, (112, 88, 86), (470, 550, 280, 14))
    for x in range(480, 720, 16):
        pygame.draw.line(bg, (90, 70, 72), (x, 522), (x, 550), 3)
    pygame.draw.line(bg, (90, 70, 72), (480, 524), (740, 524), 3)
    for i, x in enumerate((500, 590, 660)):
        pygame.draw.rect(bg, (70, 55, 60), (x - 4, 596, 64, 88))
        pygame.draw.rect(bg, (60, 48, 56) if i == 0 else (255, 226, 140),
                         (x, 600, 56, 80))
        pygame.draw.line(bg, (70, 55, 60), (x + 28, 600), (x + 28, 680), 3)

    # ground
    vgradient(bg, (0, HORIZON, CW, CH - HORIZON),
              [(0, (178, 140, 116)), (1, (148, 134, 124))])
    for _ in range(1200):
        x, y = rnd.randint(0, CW - 1), rnd.randint(HORIZON, CH - 1)
        c = bg.get_at((x, y))
        d = rnd.randint(-14, 14)
        bg.set_at((x, y), tuple(max(0, min(255, v + d)) for v in c[:3]))

    # score billboard
    for px in (110, 390):
        pygame.draw.rect(bg, (50, 45, 60), (px, 610, 14, 140))
    pygame.draw.rect(bg, (25, 25, 35), (30, 322, 450, 14))
    pygame.draw.rect(bg, (30, 30, 40), (40, 330, 430, 290), border_radius=6)
    pygame.draw.rect(bg, (8, 10, 16), (56, 346, 398, 258))
    for cx, cy, w, h in [(130, 752, 90, 26), (230, 758, 60, 20), (60, 760, 70, 22)]:
        pygame.draw.ellipse(bg, (60, 48, 52), er(cx, cy, w, h))

    # landing pad
    pygame.draw.ellipse(bg, (245, 240, 235), er(PAD_X, PAD_Y, 820, 210), 6)
    for k in range(0, 360, 10):
        t0, t1 = math.radians(k), math.radians(k + 5)
        pygame.draw.line(
            bg, (236, 206, 92),
            (PAD_X + 350 * math.cos(t0), PAD_Y + 86 * math.sin(t0)),
            (PAD_X + 350 * math.cos(t1), PAD_Y + 86 * math.sin(t1)), 6)
    pygame.draw.ellipse(bg, (150, 142, 138), er(PAD_X, PAD_Y, 620, 150), 10)
    pygame.draw.ellipse(bg, (200, 158, 122), er(PAD_X, PAD_Y, 440, 110))
    pygame.draw.ellipse(bg, (245, 240, 235), er(PAD_X, PAD_Y, 440, 110), 5)
    pygame.draw.line(bg, (245, 240, 235), (PAD_X - 120, PAD_Y), (PAD_X + 120, PAD_Y), 4)
    pygame.draw.line(bg, (245, 240, 235), (PAD_X, PAD_Y - 32), (PAD_X, PAD_Y + 32), 4)

    # traffic cone
    pygame.draw.rect(bg, (190, 80, 30), (534, 925, 52, 8))
    pygame.draw.polygon(bg, (245, 110, 40), [(540, 925), (580, 925), (560, 878)])
    pygame.draw.polygon(bg, WHITE, [(547, 906), (573, 906), (569, 894), (551, 894)])
    return bg


def make_rocket():
    """160x300 sprite; sprite centre = (80,150)."""
    s = pygame.Surface((160, 300), pygame.SRCALPHA)
    white, shade, dark, gold = (246, 246, 250), (205, 208, 222), (40, 42, 56), (236, 190, 80)

    for sx in (-1, 1):  # legs
        top, knee, foot = (80 + sx * 30, 205), (80 + sx * 62, 245), (80 + sx * 74, 284)
        pygame.draw.line(s, (220, 185, 100), top, knee, 8)
        pygame.draw.line(s, (120, 125, 140), knee, foot, 7)
        pygame.draw.ellipse(s, dark, (foot[0] - 14, foot[1] - 4, 28, 10))
        pygame.draw.rect(s, (95, 100, 120), (80 + sx * 52 - 7, 120, 14, 36), border_radius=4)

    pygame.draw.ellipse(s, white, (46, 18, 68, 190))
    pygame.draw.rect(s, white, (46, 110, 68, 120))
    pygame.draw.rect(s, shade, (98, 110, 16, 120))
    pygame.draw.ellipse(s, (255, 255, 255), (56, 40, 14, 50))

    for row in range(2):  # checker band
        for i in range(9):
            c = dark if (i + row) % 2 == 0 else white
            pygame.draw.rect(s, c, (46 + i * 7.6, 96 + row * 8, 8, 8))
    pygame.draw.rect(s, gold, (44, 118, 72, 8))
    pygame.draw.rect(s, dark, (44, 205, 72, 10))
    pygame.draw.polygon(s, dark, [(62, 230), (98, 230), (92, 248), (68, 248)])

    pygame.draw.circle(s, (150, 155, 170), (80, 160), 17)
    pygame.draw.circle(s, (20, 30, 70), (80, 160), 13)
    pygame.draw.circle(s, WHITE, (74, 154), 4)
    pygame.draw.line(s, (130, 135, 150), (80, 18), (80, 4), 3)
    pygame.draw.circle(s, (255, 130, 40), (80, 4), 4)
    return s


PIVOT_OFFSET = 20  # pivot is 20px below sprite centre (body centre)
FEET_BELOW_PIVOT = 114


def to_screen(lx, ly, piv, theta):
    x, y = lx, ly - PIVOT_OFFSET
    c, s = math.cos(theta), math.sin(theta)
    return (piv[0] + x * c + y * s, piv[1] - x * s + y * c)


def draw_flame(canvas, pts_local, piv, theta, color):
    pygame.draw.polygon(canvas, color, [to_screen(x, y, piv, theta) for x, y in pts_local])


# ---------------------------------------------------------------- main loop
def main():
    args = parse_args()
    pygame.init()
    win = pygame.display.set_mode((int(CW * args.scale), int(CH * args.scale)))
    pygame.display.set_caption("PPO Lunar Lander - AI Rocket Landing")
    clock = pygame.time.Clock()
    canvas = pygame.Surface((CW, CH))

    F = lambda size, names="arialblack,impact,arial": pygame.font.SysFont(names, size, bold=True)
    f_title, f_big, f_score = F(46), F(44), F(118)
    f_label, f_cap, f_small = F(30), F(40), F(22, "arial,helvetica")

    bg, rocket = build_background(), make_rocket()
    env = gym.make("LunarLander-v3")
    model = PPO.load(args.model)

    writer = None
    if args.record:
        try:
            import imageio
            writer = imageio.get_writer(args.out, fps=FPS, macro_block_size=1)
        except ImportError:
            print("Recording needs: pip install imageio imageio-ffmpeg")
            sys.exit(1)

    results = []
    smoke, sparks = [], []
    title_lines = args.title.split("|")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=ep)
        total, steps, action = 0.0, 0, 0
        done, hold, shake = False, 0, 0
        outcome, ocolor, crashed = "", WHITE, False
        smoke.clear()
        sparks.clear()

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    if writer:
                        writer.close()
                    env.close()
                    pygame.quit()
                    return

            # ---- physics step (AI decides)
            if not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    done, hold = True, FPS * 3
                    if truncated and not terminated:
                        outcome, ocolor = "TIME OUT", AMBER
                    elif reward <= -100:
                        outcome, ocolor, crashed = "CRASHED", RED, True
                        shake = 25
                    else:
                        outcome, ocolor = ("PERFECT LANDING" if total >= 200 else "TOUCHDOWN!"), GREEN
                    results.append((ep, total, outcome))

            x_o, y_o, ang = float(obs[0]), max(float(obs[1]), 0.0), float(obs[4])
            px = PAD_X + x_o * X_SCALE
            piv = (px, PAD_Y - FEET_BELOW_PIVOT - y_o * ALT_SCALE)
            main_on = (action == 2) and not done
            nozzle = to_screen(0, 100, piv, ang)

            if main_on:
                for _ in range(2):
                    smoke.append(dict(x=nozzle[0] + random.uniform(-8, 8), y=nozzle[1] + 20,
                                      vx=random.uniform(-1.5, 1.5), vy=random.uniform(2.5, 5),
                                      life=0, max=random.randint(30, 50), r=random.uniform(8, 14)))
            if crashed and done and hold == FPS * 3:
                for _ in range(90):
                    a, sp = random.uniform(0, 2 * math.pi), random.uniform(2, 10)
                    sparks.append(dict(x=piv[0], y=piv[1] + 60, vx=math.cos(a) * sp,
                                       vy=math.sin(a) * sp - 3, life=0,
                                       max=random.randint(40, 75), r=random.uniform(4, 11),
                                       col=random.choice([(255, 210, 80), (255, 140, 40),
                                                          (255, 250, 220), (200, 80, 40)])))

            # ---- draw
            canvas.blit(bg, (0, 0))

            # billboard text
            lab = f_label.render("SCORE", True, WHITE)
            canvas.blit(lab, (86, 366))
            scol = GREEN if total >= 0 else RED
            num = f_score.render(f"{total:+.0f}", True, scol)
            canvas.blit(num, num.get_rect(center=(255, 472)))
            pygame.draw.line(canvas, scol, (90, 556), (420, 556), 4)

            # title
            for i, line in enumerate(title_lines):
                canvas.blit(spaced(f_title, line, AMBER, 7), (50, 100 + i * 58))
            canvas.blit(f_small.render(f"EPISODE {ep}/{args.episodes}   |   STEP {steps}", True,
                                       (235, 225, 240)), (52, 106 + len(title_lines) * 58))
            pill = pygame.Rect(CW - 220, 52, 180, 40)
            pygame.draw.rect(canvas, (20, 24, 40), pill, border_radius=20)
            pygame.draw.circle(canvas, GREEN, (pill.x + 22, pill.centery), 6)
            canvas.blit(f_small.render("PPO AGENT", True, WHITE), (pill.x + 40, pill.y + 9))

            # shadow
            alt_px = y_o * ALT_SCALE
            sw, sa = max(70, int(190 - alt_px * 0.18)), max(35, int(120 - alt_px * 0.1))
            sh = pygame.Surface((sw, int(sw * 0.22)), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (30, 20, 30, sa), sh.get_rect())
            canvas.blit(sh, sh.get_rect(center=(px, PAD_Y + 8)))

            # smoke
            for p in smoke[:]:
                p["x"] += p["vx"]; p["y"] += p["vy"]; p["vx"] *= 0.98
                p["life"] += 1; p["r"] += 0.35
                if p["life"] >= p["max"]:
                    smoke.remove(p)
                    continue
                a = int(150 * (1 - p["life"] / p["max"]))
                r = int(p["r"])
                s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (225, 205, 190, a), (r, r), r)
                canvas.blit(s, (p["x"] - r, p["y"] - r))

            # flames (behind rocket)
            if not crashed:
                if main_on:
                    L = random.uniform(110, 160)
                    draw_flame(canvas, [(-16, 96), (16, 96), (10, 96 + L * .55), (0, 96 + L),
                                        (-10, 96 + L * .55)], piv, ang, (255, 140, 40))
                    draw_flame(canvas, [(-8, 96), (8, 96), (5, 96 + L * .35), (0, 96 + L * .6),
                                        (-5, 96 + L * .35)], piv, ang, (255, 235, 150))
                if action == 1 and not done:
                    L = random.uniform(50, 85)
                    draw_flame(canvas, [(-58, -22), (-58, -4), (-58 - L, -13)], piv, ang, (255, 150, 50))
                if action == 3 and not done:
                    L = random.uniform(50, 85)
                    draw_flame(canvas, [(58, -22), (58, -4), (58 + L, -13)], piv, ang, (255, 150, 50))

                rot = pygame.transform.rotate(rocket, math.degrees(ang))
                canvas.blit(rot, rot.get_rect(center=to_screen(0, 0, piv, ang)))

            # explosion
            for p in sparks[:]:
                p["x"] += p["vx"]; p["y"] += p["vy"]; p["vy"] += 0.2; p["life"] += 1
                if p["life"] >= p["max"]:
                    sparks.remove(p)
                    continue
                r = max(1, int(p["r"] * (1 - p["life"] / p["max"])))
                pygame.draw.circle(canvas, p["col"], (int(p["x"]), int(p["y"])), r)

            # subtitles
            if done:
                outlined(canvas, f_big, outcome, ocolor, (CW // 2, 1100), 4)
                outlined(canvas, f_cap, f"Final score {total:+.0f}", WHITE, (CW // 2, 1160))
            else:
                outlined(canvas, f_cap, CAPTIONS[action], WHITE, (CW // 2, 1130))

            # ---- present
            win.fill((0, 0, 0))
            view = pygame.transform.smoothscale(canvas, win.get_size())
            dx = random.randint(-8, 8) if shake > 0 else 0
            dy = random.randint(-8, 8) if shake > 0 else 0
            shake = max(0, shake - 1)
            win.blit(view, (dx, dy))
            pygame.display.flip()

            if writer:
                writer.append_data(np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)))
            clock.tick(FPS)

            if done:
                hold -= 1
                if hold <= 0:
                    break

    if writer:
        writer.close()
        print(f"Saved recording to {args.out}")
    env.close()
    pygame.quit()
    if results:
        print(f"Average reward over {len(results)} episodes: "
              f"{np.mean([r[1] for r in results]):.2f}")


if __name__ == "__main__":
    main()