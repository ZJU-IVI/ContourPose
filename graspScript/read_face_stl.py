import colorsys
import numpy as np
import itertools
from glob import glob
import os


class stl_model(object):

    def __init__(self, path):

        SUPPORTED_EXTENSIONS = ["STL"]
        self.path = list(
            itertools.chain.from_iterable(glob(os.path.join(path, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
        self.path1 = self.path[0]
        self.model1 = self.read_file(self.path1)
        self.tri = self.creat_triangles(self.model1, 0)

        for i in range(1, len(self.path)):
            self.path2 = self.path[i]
            self.model2 = self.read_file(self.path2)
            self.tri2 = self.creat_triangles(self.model2, i / float(len(self.path)))
            self.tri = self.cat_tri(self.tri, self.tri2)

    def read_file(self, path):

        normal = []
        vertex = []

        with open(path, 'r') as f:

            while True:

                p = f.readline().strip()
                if p == 'endsolid':
                    break
                word = p.split()
                if word[0] == 'facet' and word[1] == 'normal':
                    x = float(word[2])
                    y = float(word[3])
                    z = float(word[4])
                    normal.append(None)
                    normal[len(normal) - 1] = (x, y, z)
                elif word[0] == 'vertex':
                    x = float(word[1]) / 1000.
                    y = float(word[2]) / 1000.
                    z = float(word[3]) / 1000.
                    vertex.append(None)
                    vertex[len(vertex) - 1] = (x, y, z)

        assert len(normal) == len(vertex) / 3

        return {"normal": normal, "vertex": vertex}

    def creat_triangles(self, model, color):

        normal = model['normal']
        vertex = model['vertex']

        tri_num = len(normal)

        nor_list = list(set(normal))
        nor_num = len(nor_list)

        special_colors = colorsys.hsv_to_rgb(color, 1., 1.)
        triangles = []

        for i in range(tri_num):
            nor = normal[i]
            p0 = vertex[i * 3]
            p1 = vertex[i * 3 + 1]
            p2 = vertex[i * 3 + 2]
            c = special_colors
            triangles.append(None)
            triangles[len(triangles) - 1] = {"normal": nor, "p0": p0, "p1": p1, "p2": p2, "colors": c}

        return triangles

    def cat_tri(self, tri1, tri2):

        triangles = tri1 + tri2
        return triangles