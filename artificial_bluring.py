import numpy as np

from skimage.io import imread
from skimage import img_as_float
from math import ceil
from scipy.signal import convolve2d, fftconvolve

"""
Trajectory and PSFs are taken from https://github.com/KupynOrest/DeblurGAN
"""

class Trajectory(object):
    def __init__(self, canvas=64, iters=2000, max_len=80, expl=None):
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012]. Each
        trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration,
        is affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the
        previous particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming
        at inverting the particle velocity may arises, mimicking a sudden movement that occurs when the user presses
        the camera button or tries to compensate the camera shake. At each step, the velocity is normalized to
        guarantee that trajectories corresponding to equal exposures have the same length. Each perturbation (
        Gaussian, inertial, and impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi
        2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs
        :param canvas: size of domain where our trajectory os defined.
        :param iters: number of iterations for definition of our trajectory.
        :param max_len: maximum length of our trajectory.
        :param expl: this param helps to define probability of big shake. Recommended expl = 0.005.
        :param path_to_save: where to save if you need.
        """
        self.canvas = canvas
        self.iters = iters
        self.max_len = max_len
        if expl is None:
            self.expl = 0.4 * np.random.uniform(0, 1)
        else:
            self.expl = expl
            
        self.tot_length = None
        self.big_expl_count = None
        self.x = None
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]

    def fit(self):
        """
        Generate motion, you can save or plot, coordinates of motion you can find in x property.
        Also you can fin properties tot_length, big_expl_count.
        :param show: default False.
        :param save: default False.
        :return: x (vector of motion).
        """
        tot_length = 0
        big_expl_count = 0
        # how to be near the previous position
        # TODO: I can change this paramether for 0.1 and make kernel at all image
        centripetal = 0.7 * np.random.uniform(0, 1)
        # probability of big shake
        prob_big_shake = 0.2 * np.random.uniform(0, 1)
        # term determining, at each sample, the random component of the new direction
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle = 360 * np.random.uniform(0, 1)

        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl

        x = np.array([complex(real=0, imag=0)] * (self.iters))

        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=ceil((self.canvas - max(x.real)) / 2), imag=ceil((self.canvas - max(x.imag)) / 2))

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x

        return self
    
class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit()
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1)
                
                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))

        for eachPCF in range(len(self.PSFs)):
            self.PSFs[eachPCF] = self.PSFs[eachPCF] / np.sum(self.PSFs[eachPCF]+1e-15)
            
        return self.PSFs
    
def convolveRGB(image, kernel, fft=True):
    img = image.copy()
    
    if fft:
        padding_size_x, padding_size_y  = int((kernel.shape[0] - 1) / 2), int((kernel.shape[1] - 1) / 2)

        padded = np.stack([np.pad(img[...,0], pad_width=(padding_size_x,padding_size_y), mode='wrap'), 
                           np.pad(img[...,1], pad_width=(padding_size_x,padding_size_y), mode='wrap'),
                           np.pad(img[...,2], pad_width=(padding_size_x,padding_size_y), mode='wrap')], axis=2)

        img[...,0] = fftconvolve(padded[...,0], kernel, mode='valid')
        img[...,1] = fftconvolve(padded[...,1], kernel, mode='valid')
        img[...,2] = fftconvolve(padded[...,2], kernel, mode='valid')
        
    else:
        img[...,0] = convolve2d(img[...,0], kernel, mode='same', boundary='wrap')
        img[...,1] = convolve2d(img[...,1], kernel, mode='same', boundary='wrap')
        img[...,2] = convolve2d(img[...,2], kernel, mode='same', boundary='wrap')
    
    return img_as_float(img)

def blur_img(img, kernel_size=31, fft=True, argsTrajectory={}, argsPSF={}):
    
    trajectory = None
    
    if isinstance(img, str):
        img = img_as_float(imread(img))
    
    if not len(argsTrajectory):
        trajectory = Trajectory(**argsTrajectory).fit()
    
    kernel = PSF(canvas=kernel_size, fraction=[1], **argsPSF).fit()[-1]
    
    blurred = convolveRGB(image=img.copy(), kernel=kernel.copy())
    
    return blurred

