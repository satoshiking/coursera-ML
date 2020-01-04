from itertools import combinations
import matplotlib.pyplot as plt

def unique_triangle(x,y,z, triangles):
    if ((x[1] / x[0] == y[1] / y[0]) and (x[1] / x[0] == z[1] / z[0]) and (y[1] / y[0] == z[1] / z[0])):
        return False

    xs, ys, zs = sorted([x,y,z], key=lambda x: (x[0], x[1]))[0:4]
    for triangle in triangles:
        xt, yt, zt = sorted(triangle, key=lambda x: (x[0], x[1]))[0:4]
        if (xs, ys, zs) ==  (xt, yt, zt):
            return False
    return True

unique_triangle([3, -4], [-3, -5], [4, 1], [])

def count_col_triang(input_):
    points = len(input_)
    colors = set([y for (x,y) in input_])
    colors_num = len(colors)

    vertices = []
    triangles = []
    for color in colors:
        vertices.append((color, [x for (x,y) in input_ if y==color]))
        combos = list(combinations([x for (x,y) in input_ if y==color], 3))
        triangles.append((color, [(x,y,z) for (x,y,z) in combos if unique_triangle(x,y,z,combos)]))

    for my_set in triangles:
        color = my_set[0][0]
        for trng in my_set[1]:
            x = [x[0] for x in trng] + [trng[0][0]]
            y = [x[1] for x in trng] + [trng[0][1]]
            plt.plot(x, y, color=color)



    print(triangles)
    triangles_num = sorted([(x,len(y)) for (x,y) in triangles], key=lambda x: (x[1], x[0]), reverse=True)
    print(triangles_num)
    triangles_total = sum([y for (x,y) in triangles_num])
    plt.show()

    if len(triangles_num) > 0:
        colors_with_max_triangles = [x for (x,y) in triangles_num if y==triangles_num[0][1]]
        colors_with_max_triangles.append(triangles_num[0][1])
    else:
        colors_with_max_triangles = []

    return [points, colors_num, triangles_total, colors_with_max_triangles]



print(count_col_triang([
                        [[3, -4], 'blue'],
                        [[-7, -1], 'red'],
                        [[7, -6], 'yellow'],
                        [[2, 5], 'yellow'],
                        [[1, -5], 'red'],
                        [[1, 1], 'red'],
                        [[1, 7], 'red'],
                        [[1, 4], 'red'],
                        [[-3, -5], 'blue'],
                        [[4, 1], 'blue']],
                       ))
