# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


class Quaternion:
    """Quaternions for 3D rotations"""
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        
    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternion from unit vector v and rotation angle theta
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))

        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.x[:, None] * other.x

        return self.__class__([(prod[0, 0] - prod[1, 1]
                                 - prod[2, 2] - prod[3, 3]),
                                (prod[0, 1] + prod[1, 0]
                                 + prod[2, 3] - prod[3, 2]),
                                (prod[0, 2] - prod[1, 3]
                                 + prod[2, 0] + prod[3, 1]),
                                (prod[0, 3] + prod[1, 2]
                                 - prod[2, 1] + prod[3, 0])])

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        # compute theta
        norm = np.sqrt((self.x ** 2).sum(0))
        theta = 2 * np.arccos(self.x[0] / norm)

        # compute the unit vector
        v = np.array(self.x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()
        c = np.cos(theta)
        s = np.sin(theta)

        return np.array([[v[0] * v[0] * (1. - c) + c,
                          v[0] * v[1] * (1. - c) - v[2] * s,
                          v[0] * v[2] * (1. - c) + v[1] * s],
                         [v[1] * v[0] * (1. - c) + v[2] * s,
                          v[1] * v[1] * (1. - c) + c,
                          v[1] * v[2] * (1. - c) - v[0] * s],
                         [v[2] * v[0] * (1. - c) - v[1] * s,
                          v[2] * v[1] * (1. - c) + v[0] * s,
                          v[2] * v[2] * (1. - c) + c]])

class CubeAxes(plt.Axes):
    """An Axes for displaying a 3D cube"""
    # fiducial face is perpendicular to z at z=+1
    one_face = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]])

    # construct six rotators for the face
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi, 0)]
    
    # colors of the faces
    colors = ['blue', 'green', 'white', 'yellow', 'orange', 'red']
    
    def __init__(self, fig, rect=[0, 0, 1, 1], *args, **kwargs):
        # We want to set a few of the arguments
        kwargs.update(dict(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), frameon=False,
                           xticks=[], yticks=[], aspect='equal'))
        super(CubeAxes, self).__init__(fig, rect, *args, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())
        
        # define the current rotation
        self.current_rot = Quaternion.from_v_theta((1, 1, 0), np.pi / 6)
        
    
    def draw_cube(self):
        """draw a cube rotated by theta around the given vector"""
        # rotate the six faces
        Rs = [(self.current_rot * rot).as_rotation_matrix() for rot in self.rots]
        faces = [np.dot(self.one_face, R.T) for R in Rs]
        
        # project the faces: we'll use the z coordinate
        # for the z-order
        faces_proj = [face[:, :2] for face in faces]
        zorder = [face[:4, 2].sum() for face in faces]
        
        # create the polygons if needed.
        # if they're already drawn, then update them
        if not hasattr(self, '_polys'):
            self._polys = [plt.Polygon(faces_proj[i], fc=self.colors[i],
                                       alpha=0.9, zorder=zorder[i])
                           for i in range(6)]
            for i in range(6):
                self.add_patch(self._polys[i])
        else:
            for i in range(6):
                self._polys[i].set_xy(faces_proj[i])
                self._polys[i].set_zorder(zorder[i])
                
        self.figure.canvas.draw()

# 控制动画
def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)



def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def animation_quanternions(q_array):
    fig = plt.figure()
    ax = CubeAxes(fig)
    fig.add_axes(ax)

    def __animate(i):
        ax.current_rot = q_array[i]
        ax.draw_cube()
        return ax,
    anim = animation.FuncAnimation(fig, __animate, frames=len(q_array), interval=1000)
    return display_animation(anim)


def plotQuaternion(q):
    fig = plt.figure()
    ax = CubeAxes(fig)
    fig.add_axes(ax)
    ax.current_rot = q
    ax.draw_cube()


def skewV(v):
    v0=v[0]; v1=v[1]; v2=v[2];
    return np.array([
            [0, - v2, v1],
            [v2, 0, -v0],
            [-v1, v0, 0]
        ])

def overline(v):

    return np.append(0, v)

def functionL(q):
    p0 = q[0]; pv = q[1:]
    r = np.zeros((4, 4))
    r[0, :] = -q
    r[:, 0] = q
    r[1:, 1:] = p0 * np.eye(3) + skewV(pv)
    return r


def functionR(q):
    p0 = q[0]; pv = q[1:]
    r = np.zeros((4, 4))
    r[0, :] = -q
    r[:, 0] = q
    r[1:, 1:] = p0 * np.eye(3) - skewV(pv)
    return r


def dQqdq(q):
    u"""四元数微分"""
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3]
    Q0 = 2 * np.array([
        [2*q0, -q3, q2],
        [q3, 2*q0, -q1],
        [-q2, q1, 2*q0]
        ])
    Q1 = 2 * np.array([
        [2*q1, q2, q3],
        [q2, 0, -q0],
        [q3, q0, 0]
        ])
    Q2 = 2 * np.array([
        [0, q1, q0],
        [q1, 2*q2, q3],
        [-q0, q3, 0]
        ])
    Q3 = 2 *np.array([
        [0, -q0, q1],
        [q0, 0, q2],
        [q1, q2, 2*q3]
        ])
    return Q0, Q1, Q2, Q3


def S(w):
    u"""角速度的4x4斜对称矩阵"""
    wx = w[0]; wy = w[1]; wz = w[2]
    return np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])


def hatS(q):
    u"""四元数转为4x3的斜对称矩阵"""
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3]
    return np.array([
        [-q1, -q2, -q3],
        [q0, -q3, q2],
        [q3, q0, -q1],
        [-q2, q1, q0]
        ])

def Qq(q):
    u"""将四元数转换为旋转矩阵"""
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3]
    return np.array([
        [2*q0*q0-1+2*q1*q1, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q3],
        [2*q1*q2 + 2*q0*q3, 2*q0*q0-1+2*q2*q2, 2*q1*q3 - 2*q0*q3],
        [2*q1*q3 + 2*q0*q2, 2*q2*q3 + 2*q0*q1, 2*q0*q0-1+2*q3*q3]
        ])


def expq(v):
    norm2 = np.linalg.norm(v, ord=2)
    return np.append(np.cos(norm2), v / norm2 * np.sin(norm2))


def expR(v):
    norm2 = np.linalg.norm(v, ord=2)
    return np.eye(3) + np.sin(norm2) * skewV(v / norm2) + (1 - np.cos(norm2)) * np.dot(skewV(v / norm2), skewV(v / norm2))

def circledot(p, q):
    return np.dot(functionL(p), q)

def norm_q(q):
    q = q / np.linalg.norm(q)
    q *= (1 if q[0] > 0 else -1)
    return q


def inverse_q(q):
    return np.array([q[0], - q[1], -q[2], -q[3]])
