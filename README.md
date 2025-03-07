import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_square_points(n):
    """توليد نقاط على شكل مربع"""
    side = int(np.sqrt(n))
    x = np.linspace(-1, 1, side)
    y = np.linspace(-1, 1, side)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    return points[:n]  # لضمان العدد المطلوب من النقاط

def generate_circle_points(n):
    """توليد نقاط على شكل دائرة"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack([x, y])

def animate(i, points, start, end, scatter):
    """تحديث مواضع النقاط تدريجياً"""
    t = i / frames  # نسبة التقدم
    points[:, 0] = (1 - t) * start[:, 0] + t * end[:, 0]
    points[:, 1] = (1 - t) * start[:, 1] + t * end[:, 1]
    scatter.set_offsets(points)
    return scatter,

# إعداد البيانات
n_points = 100  # عدد النقاط
frames = 50  # عدد الإطارات
square_points = generate_square_points(n_points)
circle_points = generate_circle_points(n_points)
points = np.copy(square_points)  # البداية من المربع

# إنشاء الشكل
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
scatter = ax.scatter(points[:, 0], points[:, 1])

# إنشاء الحركة
ani = animation.FuncAnimation(fig, animate, frames=frames, fargs=(points, square_points, circle_points, scatter), interval=50)
plt.show()
