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

shutil.copy2("logs/op-mpc-x/op-mpc-x.csv", f"logs/op-mpc-x/op-mpc-x_{d}.csv")

data_set = np.loadtxt(
    fname="logs/op-mpc-x/op-mpc-x.csv", #読み込むファイルのパスと名前
    dtype="float", #floatで読み込む
    delimiter=",", #csvなのでカンマで区切る
)

DT = data_set[1, 0] - data_set[0, 0]

# print(data_set)

# 1列目が time
# 2列目が u
# 3~7列目が x[0], x[1], x[2], x[3]

fig, (ax, ax2) = plt.subplots(2, 1)

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
ax2.legend(loc='upper left')
ax2_2.legend(loc='upper right')
# h2, l2 = ax2.get_legend_handles_labels()
# h2_2, l2_2 = ax2_2.get_legend_handles_labels()
# ax2.legend(h2 + h2_2, l2 + l2_2)

# ax2に縦線を描画しアニメーションと一致させる
line = ax2.axvline(x=0, color='k', linestyle=':')

scale = 1
# xlim_max = data_set[:, 6].max() if data_set[:, 6].max() > scale else scale
# xlim_min = data_set[:, 6].min() if data_set[:, 6].min() < -scale else -scale
xlim_max = data_set[:, 2].max() if data_set[:, 2].max() > scale else scale
xlim_min = data_set[:, 2].min() if data_set[:, 2].min() < -scale else -scale
ax.set_xlim(xlim_min-1, xlim_max+1)
ax.set_ylim(-r, 0.5)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_aspect('equal')
# ax.set_title('v0={},theta={}°'.format(v0,theta*180/np.pi))


# anim = FuncAnimation(fig, update_anim, frames=np.arange(0,len(t)),interval=interval,blit=True,repeat=True)

# https://note.nkmk.me/python-matplotlib-patches-circle-rectangle/
c = patches.Circle(xy=(0, 0), radius=r, fc='None', ec='k')
# e = patches.Ellipse(xy=(-0.25, 0), width=0.5, height=0.25, fc='b', ec='y')
rect = patches.Rectangle(xy=(1, 1), width=W, height=W, angle=45, rotation_point='center', ec='k', fill=False)
# ax.add_patch(e)
con = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='k', fc='w', label="act")
ax.add_patch(c)
ax.add_patch(con)
ax.add_patch(rect)

con_est = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='b', fc='w', linestyle=':', label="est")
ax.add_patch(con_est)
con_pred = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='orange', fc='w', linestyle=':', label="pred")
ax.add_patch(con_pred)
con_ref = patches.ConnectionPatch((0, 0), (1, 1), coordsA='data', linewidth=3, ec='r', fc='w', linestyle=':', label="ref")
ax.add_patch(con_ref)

ax.legend()

# def update_anim(frame_num):
#     obj1.set_data(v0*np.cos(theta)*t[frame_num],y1.T[0][frame_num]) #(水平方向の速度×経過時間, 鉛直方向の位置)
#     obj2.set_data(np.cos(theta),y2.T[0][frame_num])
#     return obj1, obj2,

def update_anim(step, _step_max):
    x = data_set[step, 2]
    theta = data_set[step, 4]
    X = x + L * np.sin(theta)
    Y = L * np.cos(theta)
    # print(x, theta, X, Y)
    c.center = (x, 0)
    rect.set_xy((X-W/2, Y-W/2))
    rect.set_angle(-theta * 180 / np.pi)
    con.xy1 = x, 0
    con.xy2 = X, Y
    line.set_xdata([data_set[step, 0], data_set[step, 0]])

    theta_est = data_set[step, 8]
    con_est.xy1 = data_set[step, 6], 0
    con_est.xy2 = data_set[step, 6] + L * np.sin(theta_est), L * np.cos(theta_est)

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
# ani = FuncAnimation(fig, update_anim, fargs=(len(data_set),), interval=interval, frames=len(data_set), repeat=True)
ani = FuncAnimation(fig, update_anim, fargs=(len(data_set),), interval=interval, frames=len(data_set), repeat=False)

# ani.save(f"imgs/anim_{d}.gif", writer="pillow")
# ani.save(f"imgs/anim_{d}.gif", writer="imagemagick")


# ffmpegが必要

# plt.show()

path = f"imgs/anim_{d}.mp4"
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

# ani.save(f"imgs/anim_{d}.gif", writer="pillow")
