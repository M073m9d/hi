import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment

def generate_square_border_points(n, num_lines=4):
    """توليد نقاط على حدود المربع فقط مع تقسيمها إلى خطوط"""
    side = n // num_lines  # توزيع النقاط على عدد الخطوط المطلوبة
    x = np.linspace(-1, 1, side)
    y = np.linspace(-1, 1, side)
    top = np.column_stack([x, np.ones_like(x)])
    bottom = np.column_stack([x, -np.ones_like(x)])
    left = np.column_stack([-np.ones_like(y), y])
    right = np.column_stack([np.ones_like(y), y])
    points = np.vstack([top, bottom, left, right])
    return points[:n], [top, bottom, left, right]  # إرجاع النقاط والخطوط

def generate_circle_border_points(n, num_lines=4):
    """توليد نقاط على محيط الدائرة مع تقسيمها إلى خطوط"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack([x, y])
    return points, np.array_split(points, num_lines)  # تقسيم النقاط إلى خطوط

def match_points(start, end):
    """استخدام خوارزمية Hungarian لإيجاد أفضل تطابق بين النقاط"""
    cost_matrix = np.linalg.norm(start[:, np.newaxis] - end, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return end[col_ind]

def get_bezier_curve(p0, p1, num_points=100, amplitude=0.5):
    """
    حساب منحنى Bezier من الدرجة الثانية باستخدام نقطة البداية والنهاية ونقطة تحكم محسوبة
    بحيث يكون المنحنى منحنيًا بشكل واضح.
    """
    mid = (p0 + p1) / 2
    diff = p1 - p0
    norm = np.linalg.norm(diff)
    if norm == 0:
        norm = 1
    perp = np.array([-diff[1], diff[0]]) / norm
    control = mid + perp * amplitude
    t = np.linspace(0, 1, num_points)[:, None]
    curve = (1 - t)**2 * p0 + 2 * (1 - t) * t * control + t**2 * p1
    return curve

def animate(i, start_lines, end_lines, scatter, num_lines, frames_per_line):
    """
    تحريك كل خط بشكل تسلسلي:
      - لكل خط فترة زمنية خاصة به يبدأ بعدها الحركة.
      - خلال فترة الحركة يتم استخدام تأثير ease in/out.
    """
    new_points_lines = []
    for j, (start, end) in enumerate(zip(start_lines, end_lines)):
        start_frame = j * frames_per_line
        end_frame = (j + 1) * frames_per_line
        
        if i < start_frame:
            f = 0
        elif i >= end_frame:
            f = 1
        else:
            f = (i - start_frame) / (end_frame - start_frame)
            f = (1 - np.cos(f * np.pi)) / 2  # تأثير ease in/out
        
        new_pos = start + (end - start) * f
        new_points_lines.append(new_pos)
    
    all_points = np.vstack(new_points_lines)
    scatter.set_offsets(all_points)
    return scatter,

# إعداد البيانات
n_points = 100   # عدد النقاط
num_lines = 4    # تقسيم التشكيل إلى خطوط
frames = 80      # إجمالي عدد الإطارات (يمكن تعديلها)
frames_per_line = frames // num_lines  # عدد الإطارات لكل خط

# توليد نقاط الحدود للمربع والدائرة
square_points, square_lines = generate_square_border_points(n_points, num_lines)
circle_points, circle_lines = generate_circle_border_points(n_points, num_lines)

# تعيين الإزاحات بحيث يكون المربع في الإحداثيات الموجبة والدائرة في الإحداثيات السالبة
square_offset = np.array([1.5, 1.5])
circle_offset = np.array([-1.5, -1.5])
square_points = square_points + square_offset
square_lines = [line + square_offset for line in square_lines]
circle_points = circle_points + circle_offset
circle_lines = [line + circle_offset for line in circle_lines]

# تحسين توزيع النقاط لكل خط باستخدام Hungarian
circle_lines = [match_points(s, e) for s, e in zip(square_lines, circle_lines)]

# إنشاء الشكل البياني
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
scatter = ax.scatter(square_points[:, 0], square_points[:, 1])

# رسم مسارات إرشادية لكل خط بألوان مختلفة لتوضيح الحركة
colors = plt.cm.viridis(np.linspace(0, 1, num_lines))
for j, (s_line, c_line) in enumerate(zip(square_lines, circle_lines)):
    start_center = np.mean(s_line, axis=0)
    end_center = np.mean(c_line, axis=0)
    curve = get_bezier_curve(start_center, end_center, num_points=100, amplitude=0.5)
    ax.plot(curve[:, 0], curve[:, 1], '--', color=colors[j], linewidth=2, 
            label=f'خط {j+1} (دليل)')

ax.legend(loc='upper right')

# بدء الأنيميشن: حركة كل خط بشكل تسلسلي دون تداخل
ani = animation.FuncAnimation(fig, animate, frames=frames, 
                              fargs=(square_lines, circle_lines, scatter, num_lines, frames_per_line), 
                              interval=50)
plt.show()
