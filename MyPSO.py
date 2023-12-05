import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
import time

def calculate_distance(point1, point2):
    """
    计算两点之间的欧几里得距离。
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def is_in_obstacle(point, obstacles):
    """
    检查点是否在任何障碍物内。
    """
    for obstacle in obstacles:
        center, radius = obstacle
        if calculate_distance(point, center) < radius:
            return True
    return False

'''
def fitness_function(path, obstacles):
    """
    计算路径的适应度。

    参数:
    path -- 路径的表示，是一个包含多个点的列表。每个点是一个二维坐标 (x, y)。
    obstacles -- 障碍物列表，每个障碍物由中心点和半径表示。

    返回:
    fitness -- 路径的适应度值。路径越短且避开障碍物，适应度越高（数值越小）。
    """
    total_distance = 0
    obstacle_penalty = 1000  # 遇到障碍物时的惩罚值

    for i in range(1, len(path)):
        segment_distance = calculate_distance(path[i], path[i-1])
        print(str(i)+":"+str(segment_distance))
        total_distance += segment_distance

        # 检查路径的这一段是否穿过障碍物
        if is_in_obstacle(path[i], obstacles):
            total_distance += obstacle_penalty

    return total_distance

# 示例使用
obstacles = [((2, 2), 1), ((4, 4), 1.5)]  # 障碍物列表
path = [(0, 0), (1, 2), (3, 3), (5, 5),(6, 6)]  # 示例路径

fitness = fitness_function(path, obstacles)
print("Path fitness:", fitness)

'''

def fitness_function(position):
    """
    计算给定位置的适应度。

    参数:
    position -- 一个二维坐标 (x, y)。

    返回:
    fitness -- 适应度值，该值越小越好。
    """
    # x, y ,z= position
    # fitness = (x-5)**2 + (y-5)**2  # 计算二次函数的值
    # fitness = ((x-1)**2)+((y-1)**2)+((z-1)**2)
    total_distance = 0
    obstacle_penalty = 1000  # 遇到障碍物时的惩罚值
    #------------完整路径点的成本计算-------------
    total_distance += calculate_distance((-10,-10,0),position[0])
    for i in range(1, len(position)):
        segment_distance = calculate_distance(position[i], position[i-1])
        # print(str(i)+":"+str(segment_distance))
        total_distance += segment_distance
    total_distance += calculate_distance(position[len(position)-1],(10,10,10))
    #-------------
    for i in range(1, len(position)):
        if(0.5 > calculate_distance(position[i], position[i-1])):
            total_distance +=10
    #-------------障碍物碰撞检测
    obstacles = [(5,5,5),2]
    # 检查路径的这一段是否穿过障碍物
    # for i in range(0, len(position)):
    #     if is_in_obstacle(position[i], obstacles):
    #         total_distance += obstacle_penalty

    return total_distance


#===========================================================================
def update_particles(particles, velocities, p_best, g_best, bounds, w=0.5, c1=1, c2=1):
    """
    更新粒子群中每个粒子的速度和位置。

    参数:
    particles -- 粒子群的当前位置。
    velocities -- 粒子群的当前速度。
    p_best -- 每个粒子历史上的最佳位置。
    g_best -- 目前为止整个群体的最佳位置。
    bounds -- 搜索空间的边界。
    w -- 惯性权重。
    c1, c2 -- 加速系数。

    返回:
    更新后的粒子位置和速度。
    """
    n_particles, dimensions = particles.shape
    new_velocities = np.zeros((n_particles, dimensions))
    new_particles = np.zeros((n_particles, dimensions))

    for i in range(n_particles):
        for d in range(dimensions):
            r1, r2 = random.random(), random.random()
            # 更新速度
            new_velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1 * (p_best[i][d] - particles[i][d]) +
                                    c2 * r2 * (g_best[i][d] - particles[i][d])
                                    # +random.random() * 0.5
                                    # -random.random() * 0.5
                                    )
            # 应用速度更新位置
            new_particles[i][d] = (particles[i][d] + new_velocities[i][d]
                                   +random.random()*0.1
                                   -random.random()*0.1
                                   )

            # 确保粒子不会超出预定义边界
            if new_particles[i][d] < bounds[d][0]:
                new_particles[i][d] = bounds[d][0]
            if new_particles[i][d] > bounds[d][1]:
                new_particles[i][d] = bounds[d][1]

    return new_particles, new_velocities
#===========================================================================
'''
# 初始化参数
num_particles = 10  # 粒子数量
num_dimensions = 3  # 维度数，这里是二维空间
bounds = [(-10, 10), (-10, 10),(-10, 10)]  # 搜索空间的边界

# 初始化粒子群
particles  = np.random.uniform(-10, 10, (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

print(particles)
# 初始化每个粒子的最佳位置和全局最佳位置
p_best = np.copy(particles)
g_best = np.copy(p_best[np.argmin([fitness_function(p) for p in p_best])])
# p_best =
# g_best = p_best[0]
# 运行PSO算法
max_iterations = 100  # 最大迭代次数
for _ in range(max_iterations):
    # 更新粒子状态
    particles, velocities = update_particles(particles, velocities, p_best, g_best, bounds)

    # 计算新的个体最优和全局最优位置
    for i in range(num_particles):
        if fitness_function(particles[i]) < fitness_function(p_best[i]):
            p_best[i] = particles[i]
            if fitness_function(p_best[i]) < fitness_function(g_best):
                g_best = p_best[i]

# print(particles)
# print(len(p_best))
# 算法结束后，g_best 将包含找到的最佳解
print("Best position found:", g_best)

'''
#===========================================================================
# 初始化参数
num_particles = 200          # 粒子数量
dimension_per_point  = 3    # 维度数，这里是x维空间
num_points_per_path = 10     # 每条路径包含x个点
bounds = [(-10, 10), (-10, 10),(-10, 10)]  # 搜索空间的边界
# bounds = [(-10, 10), (-10, 10)]  # 搜索空间的边界

# 初始化粒子群
particles  = np.random.uniform(-10, 10, (num_particles,num_points_per_path, dimension_per_point))
velocities = np.random.uniform(-1, 1, (num_particles,num_points_per_path, dimension_per_point))

print(particles)
# 初始化每个粒子的最佳位置和全局最佳位置
p_best = np.copy(particles)
# print(p_best)
# g_best = np.copy(p_best[np.argmin([fitness_function(p) for p in p_best])])
# print('g_best:'+str(g_best))
# p_best = particles
g_best = np.copy(p_best[0])
# 运行PSO算法
max_iterations = 1000  # 最大迭代次数
# for _ in range(max_iterations):

plt.ion()  # 开启交互模式
# 创建图形和散点图
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

# 初始化散点图
x, y, z = np.random.rand(3, 10)
scat = ax.scatter(x, y, z)
# 创建样条插值
t = np.arange(len(x))
spline_x = CubicSpline(t, x)
spline_y = CubicSpline(t, y)
spline_z = CubicSpline(t, z)

# 生成细分点
tnew = np.linspace(0, t.max(), 300)
xnew = spline_x(tnew)
ynew = spline_y(tnew)
znew = spline_z(tnew)

# 绘制曲线并保存引用
line, = ax.plot(xnew, ynew, znew, color='blue')

# 设置坐标轴范围
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

lowCostPath = g_best
CostPath = -1

count=0
while True:
    count += 1
    if count >=100:
        plt.pause(0.1)
        continue
    for i in range(0,num_particles):
        # 更新粒子状态
        particles[i], velocities[i] = update_particles(particles[i],
                                                       velocities[i],
                                                       p_best[i],
                                                       g_best,
                                                       # lowCostPath,
                                                       bounds)

        # 计算新的个体最优和全局最优位置
        # for i in range(num_particles):
        if fitness_function(particles[i]) < fitness_function(p_best[i]):
            p_best[i] = particles[i]
            if fitness_function(p_best[i]) < fitness_function(g_best):
                g_best = p_best[i]
                # print("cost[",i,']', fitness_function(g_best[i]))
                if CostPath < 0 :
                    CostPath =  fitness_function(g_best)
                    lowCostPath = g_best
                if CostPath >  fitness_function(g_best):
                    CostPath = fitness_function(g_best)
                    lowCostPath = g_best

    # 算法结束后，g_best 将包含找到的最佳解
    print("cost"+'['+str(count)+']:',CostPath)
    # print(lowCostPath)

    # 清除前一次的散点图数据
    scat.remove()

    x_p, y_p,z_p = zip(*lowCostPath)
    scat = ax.scatter(x_p, y_p, z_p)

    x, y, z = lowCostPath[:, 0], lowCostPath[:, 1], lowCostPath[:, 2]
    t = np.arange(x.shape[0])
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    spline_z = CubicSpline(t, z)
    tnew = np.linspace(0, t.max(), 300)
    xnew = spline_x(tnew)
    ynew = spline_y(tnew)
    znew = spline_z(tnew)

    # 更新曲线数据
    line.set_xdata(xnew)
    line.set_ydata(ynew)
    line.set_3d_properties(znew)

    # x, y,z = zip(*particles[0])
    # scat = ax.scatter(x, y, z)
    # scat.set_offsets(np.column_stack([x_positions, y_positions,1]))

    # x1,y1,z1 = lowCostPath[0]
    # x2,y2,z2= lowCostPath[num_points_per_path-1]
    # 更新数据
    # line.set_data([x1, x2], [y1, y2])
    # line.set_3d_properties([z1, z2])
    # 重绘图形
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    plt.draw()
    plt.pause(0.01)  # 暂停一段时间以显示更新
    # time.sleep(0.01)
