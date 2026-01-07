import numpy as np

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices
        x = [a.x for a in self.vertices]
        y = [a.y for a in self.vertices]
        self.mid = Vertex(np.average(x), np.average(y))
        self.ref = self.vertices[0]

        self.edges = getEdges(self.vertices)
        self.x_range = [min(x), max(x)]
        self.y_range = [min(y), max(y)]
        for edge in self.edges:
            for vertex in self.vertices:
                edge.calibrate(vertex)


    def get(self):
        xx, yy = np.mgrid[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]
        img = np.ones([self.x_range[1]-self.x_range[0], self.y_range[1]-self.y_range[0]])
        for edge in self.edges:
            img = np.logical_and(img, edge.relPointPos(xx, yy))

        img = img.T
        return img

class Rectangle:
    def __init__(self, a, b):
        self.a = b
        self.b = a

    def get(self):
        return np.ones([self.a, self.b])

class Circle:
    def __init__(self, r):
        self.r = r

    def get(self):
        x = self.r
        xx, yy = np.mgrid[0:self.r*2, 0:self.r*2]
        return ((xx - x) ** 2 + (yy - x) ** 2) ** 0.5 < self.r

class Vertex:
    def __init__(self, px, py, c=1):
        self.x = px
        self.y = py
        self.c = c

    def arr(self):
        return np.array([self.x, self.y, self.c])

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y + self.c * other.c

    def length(self):
        return (self.x * self.x + self.y * self.y)**0.5

    def print(self):
        print(self.x, self.y)

class Edge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.x0 = v0.x
        self.y0 = v0.y
        self.x1 = v1.x
        self.y1 = v1.y
        self.inverse = False
        self.above = 0
        self.n = Vertex(*np.cross(self.v0.arr(), self.v1.arr()))


        assert self.v0 * self.n == 0
        assert self.v1 * self.n == 0

    def calibrate(self, v):
        if self.n * v > 0:
            self.above += 1
        elif self.n * v < 0:
            self.above -= 1

        if self.above >= 0:
            self.inverse = False
        else:
            self.inverse = True


    def relPointPos(self, px, py):
        v = Vertex(px, py)
        if self.inverse:
            return v * self.n <= 0
        return v * self.n >= 0

def createRandomPolygon(vertices):
    start = Vertex(*np.random.random_integers(10, 410, 2))
    vertices_ = []
    if vertices == 3:
        for a in range(3):
            vertices_.append(Vertex(*np.random.random_integers(0, 512, 2)))
    else:
        angles = 2*np.pi * np.arange(0, 360, 360/vertices) / 360
        dist = np.random.random_integers(5, min(start.x, start.y, 512 - start.x, 512 - start.y))

        for angle in angles:
            v = Vertex(start.x + int(np.sin(angle) * dist), start.y + int(np.cos(angle) * dist))
            vertices_.append(v)

    label = vertices - 1
    polygon = Polygon(vertices_)
    return polygon, label

def getAngle(mid, v0, v):
    v_rel = v - mid
    k = min(1, max(-1, (v0 * v_rel) / (v0.length() * v_rel.length())))
    if v_rel.x < 0:
        return 2*np.pi - np.arccos(k)

    return np.arccos(k)

def getEdges(vertices):
    edges = []
    for a in range(len(vertices)):
        edges.append(Edge(vertices[a], vertices[a-1]))

    return edges

def addShape(arr, shape, position):
    shape_im = shape.get()
    y0, x0 = position
    x1, y1 = shape_im.shape
    x1 = x0 + x1
    y1 = y0 + y1
    if y1 >= 512:
        y1 = 511
    if x1 >= 512:
        x1 = 511
    arr[x0:x1, y0:y1] = np.logical_or(shape_im[:x1-x0, :y1-y0], arr[x0:x1, y0:y1])
    return arr
