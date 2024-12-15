#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import datetime
import shutil
import os

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
d = now.strftime('%Y%m%d%H%M%S')
r = 0.05
L = 0.270
W = 200e-3

src = "logs/mppi/mppi.csv"
dst = f"logs/mppi/mppi_{d}.csv"
shutil.copy2(src, dst)

data_set = np.loadtxt(
    fname=dst, #読み込むファイルのパスと名前
    dtype="float", #floatで読み込む
    delimiter=",", #csvなのでカンマで区切る
)

DT = data_set[1, 0] - data_set[0, 0]

# 1列目が time
# 2列目が u
# 3~7列目が x[0], x[1], x[2], x[3]

fig, (ax, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(left=0.2, right=0.8)

# 左軸 x, x', u
# 右軸 θ, θ'

ax2_2 = ax2.twinx()
ax2.plot(data_set[:, 0], data_set[:, 1], label="u", linestyle='-.', color='r')
ax2.plot(data_set[:, 0], data_set[:, 2], label="x", color='b')
ax2.plot(data_set[:, 0], data_set[:, 3], label="x'", color='g')
ax2_2.plot(data_set[:, 0], data_set[:, 4], label="θ", linestyle='-.', color='gold')
ax2_2.plot(data_set[:, 0], data_set[:, 5], label="θ'", linestyle='-.', color='darkorange')
ax2.set_xlabel("time [s]")
ax2.set_ylabel("displacement [m, m/s, 1]")
ax2_2.set_ylabel("angle [rad, rad/s]")

ax2.legend(loc='center right', bbox_to_anchor=(-0.1, 1))
ax2_2.legend(loc='center left', bbox_to_anchor=(1.1, 1))

# ax2に縦線を描画しアニメーションと一致させる
line = ax2.axvline(x=0, color='k', linestyle=':')

scale = 1
xlim_max = data_set[:, 2].max() if data_set[:, 2].max() > scale else scale
xlim_min = data_set[:, 2].min() if data_set[:, 2].min() < -scale else -scale
ax.set_xlim(xlim_min-1, xlim_max+1)
ax.set_ylim(-r, 0.5)

c = patches.Circle(xy=(0, 0), radius=r, fc='None', ec='k')
rect = patches.Rectangle(xy=(1, 1), width=W, height=W, angle=45, rotation_point='center', ec='k', fill=False)
con = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='k', fc='w', label="act")
ax.add_patch(c)
ax.add_patch(con)
ax.add_patch(rect)

ax.legend()

def update_anim(step, _step_max):
    x = data_set[step, 2]
    theta = data_set[step, 4]
    X = x + L * np.sin(theta)
    Y = L * np.cos(theta)
    c.center = (x, 0)
    rect.set_xy((X-W/2, Y-W/2))
    rect.set_angle(-theta * 180 / np.pi)
    con.xy1 = x, 0
    con.xy2 = X, Y
    line.set_xdata([data_set[step, 0], data_set[step, 0]])

    # theta_est = data_set[step, 8]
    # con_est.xy1 = data_set[step, 6], 0
    # con_est.xy2 = data_set[step, 6] + L * np.sin(theta_est), L * np.cos(theta_est)

    # data_set の長さが13以上の場合
    if len(data_set[step]) >= 13:
        theta_pred = data_set[step, 12]
        con_pred.xy1 = data_set[step, 10], 0
        con_pred.xy2 = data_set[step, 10] + L * np.sin(theta_pred), L * np.cos(theta_pred)

    if len(data_set[step]) >= 17:
        theta_ref = data_set[step, 16]
        con_ref.xy1 = data_set[step, 14], 0
        con_ref.xy2 = data_set[step, 14] + L * np.sin(theta_pred), L * np.cos(theta_pred)

    t = data_set[step, 0]
    ax.set_title(f"step={step:4}, t={t:.3f}")

    if step >= _step_max:
        print("end")
        plt.close()

interval = DT * 1000

ani = FuncAnimation(fig, update_anim, fargs=(len(data_set),), interval=interval, frames=len(data_set), repeat=False)

path = f"imgs/anim_mppi_{d}.mp4"
s = f"<video src=\"../{path}\" controls autoplay loop></video>\n"

try:
    ani.save(path, writer="ffmpeg")
    with open('imgs/anim.md', mode='r') as reader:
        s = s + reader.read()
    with open("imgs/anim.md", mode='w') as f:
        f.write(s)
    print(f"saved: {path}")
except Exception as e:
    print(e)
    os.remove(path)

"""
xhost +local:
"""
