#!/usr/bin/env python3

"""
xhost +local:
"""

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

src = "logs/mpc-ukf.csv"
dst = f"logs/mpc-ukf-{d}.csv"
shutil.copy2(src, dst)

data_set = np.loadtxt(
    fname=dst, #読み込むファイルのパスと名前
    dtype="float", #floatで読み込む
    delimiter=",", #csvなのでカンマで区切る
)

DT = data_set[1, 0] - data_set[0, 0]

# 0列目が time
# 1列目が u
# 2列目が x[0] = x
# 3列目が x[1] = x'
# 4列目が x[2] = x''
# 5列目が x[3] = θ
# 6列目が x[4] = θ'
# 7列目が x[5] = θ''
# 8列目が x_est[0] = x_est
# 9列目が x_est[1] = x'_est
# 10列目が x_est[2] = x''_est
# 11列目が x_est[3] = θ_est
# 12列目が x_est[4] = θ'_est
# 13列目が x_est[5] = θ''_est

fig, (ax, ax2, ax3) = plt.subplots(3, 1)
fig.subplots_adjust(left=0.2, right=0.8)

# ax: 車両の位置と姿勢
# 実際の値
c = patches.Circle(xy=(0, 0), radius=r, fc='None', ec='k')
rect = patches.Rectangle(xy=(1, 1), width=W, height=W, angle=45, rotation_point='center', ec='k', fill=False)
con = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='k', fc='w', label="act")
con_est = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='b', fc='w', linestyle=':', label="est")
ax.add_patch(c)
ax.add_patch(con)
ax.add_patch(rect)
ax.add_patch(con_est)

ax.legend()

scale = 1
xlim_max = data_set[:, 2].max() if data_set[:, 2].max() > scale else scale
xlim_min = data_set[:, 2].min() if data_set[:, 2].min() < -scale else -scale
ax.set_xlim(xlim_min-1, xlim_max+1)
ax.set_ylim(-r, 0.5)

# ax2: 左軸 実値 u, x, x'

ax2.plot(data_set[:, 0], data_set[:, 1], label="u", linestyle='-.', color='r')
ax2.plot(data_set[:, 0], data_set[:, 2], label="x", color='b')
ax2.plot(data_set[:, 0], data_set[:, 3], label="x'", color='g')
ax2.set_xlabel("time [s]")
ax2.set_ylabel("displacement [m, m/s, 1]")
ax2.legend(loc='center right', bbox_to_anchor=(-0.1, 1))

# ax2_2: 右軸 実値 θ, θ'
ax2_2 = ax2.twinx()
ax2_2.plot(data_set[:, 0], data_set[:, 5], label="θ", linestyle='-.', color='gold')
ax2_2.plot(data_set[:, 0], data_set[:, 6], label="θ'", linestyle='-.', color='darkorange')
ax2_2.set_ylabel("angle [rad, rad/s]")
ax2_2.legend(loc='center left', bbox_to_anchor=(1.1, 1))

# ax2に縦線を描画しアニメーションと一致させる
line = ax2.axvline(x=0, color='k', linestyle=':')

# ax3: 左軸 推定値 u, x_est, x'_est
ax3.plot(data_set[:, 0], data_set[:, 1], label="u", linestyle='-.', color='r')
ax3.plot(data_set[:, 0], data_set[:, 8], label="x", color='b')
ax3.plot(data_set[:, 0], data_set[:, 9], label="x'", color='g')
ax3.set_xlabel("time [s]")
ax3.set_ylabel("displacement [m, m/s, 1]")
ax3.legend(loc='center right', bbox_to_anchor=(-0.1, 1))

# ax3_2: 右軸 推定値 θ_est, θ'_est
ax3_2 = ax3.twinx()
ax3_2.plot(data_set[:, 0], data_set[:, 11], label="θ", linestyle='-.', color='gold')
ax3_2.plot(data_set[:, 0], data_set[:, 12], label="θ'", linestyle='-.', color='darkorange')
ax3_2.set_ylabel("angle [rad, rad/s]")
ax3_2.legend(loc='center left', bbox_to_anchor=(1.1, 1))

# ax3に縦線を描画しアニメーションと一致させる
line2 = ax3.axvline(x=0, color='k', linestyle=':')

def update_anim(step, _step_max):
    x = data_set[step, 2]
    theta = data_set[step, 5]
    X = x + L * np.sin(theta)
    Y = L * np.cos(theta)
    c.center = (x, 0)
    rect.set_xy((X-W/2, Y-W/2))
    rect.set_angle(-theta * 180 / np.pi)
    # 実値
    con.xy1 = x, 0
    con.xy2 = X, Y

    # 推定値
    theta_est = data_set[step, 11]
    con_est.xy1 = data_set[step, 8], 0
    con_est.xy2 = data_set[step, 8] + L * np.sin(theta_est), L * np.cos(theta_est)

    line.set_xdata([data_set[step, 0], data_set[step, 0]])
    line2.set_xdata([data_set[step, 0], data_set[step, 0]])

    t = data_set[step, 0]
    ax.set_title(f"step={step:4}, t={t:.3f}")

    if step >= _step_max:
        print("end")
        plt.close()

interval = DT * 1000

ani = FuncAnimation(fig, update_anim, fargs=(len(data_set),), interval=interval, frames=len(data_set), repeat=False)

path = f"imgs/anim_mpc_ukf_{d}.mp4"
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
