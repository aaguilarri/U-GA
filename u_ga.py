import sys
import getopt as gt
import numpy as np
import matplotlib.pyplot as plt
import test_functions as tf
from FObj import FObj
import scipy.linalg
from scipy.stats import norm


class BGA:
    def __init__(self, n_b, n_p, func):
        self.nb = n_b
        self.np = n_p
        if not isinstance(func, FObj):
            print("Objective Function should be of FObj type")
            exit(-1)
        self.func = func
        self.fs = None
        self.pb = 2*np.random.rand(n_b)-1 #0.5 * np.ones(n_b)
        # self.covb = 0.01*np.ones(n_b) + 0.99 * np.eye(n_b)
        # self.covb = 1 * np.eye(n_b)
        # self.covb = 2*np.random.rand(n_b, n_b) - np.ones((n_b, n_b))
        self.covb = 0.5*np.random.rand(n_b, n_b)
        self.covb = self.covb @ self.covb.T
        self.R = np.random.rand(n_p, n_p)
        self.R = self.R.T @ self.R
        self.Q = np.random.rand(n_b, n_b)
        self.Q = self.Q.T @ self.Q
        [p1, p2] = self.get_pop()
        self.popf = p1
        self.pop = p2
        self.pb = np.mean(self.popf, axis=0)
        self.covb = np.cov(self.popf.T)
        self.dpb = 0.
        self.dcb = 0.
        # to test creating population of 2np, evaluate and choose the best ones
        # to compute pb and covb from them
        pass

    def get_pop(self):
        # print(f"self.pb. shape = {np.shape(self.pb)}")
        # covo = self.covb.T @ self.covb
        pop0 = np.random.multivariate_normal(self.pb, self.covb, self.np)
        pop1 = (np.sign(pop0 - self.pb) + 1)/2
        return [pop0, pop1]

    def predict(self):
        p0 = self.pop
        p0f = self.popf
        # print(self.covb)
        # print(self.pb)
        # print("*")
        [p1f, p1] = self.get_pop()
        f0 = self.eval(p0)
        f1 = self.eval(p1)
        ps = np.vstack((p0, p1))
        fs = np.hstack((f0, f1))
        pfs = np.vstack((p0f, p1f))
        fps = list(zip(fs, ps, pfs))
        fps.sort(key=lambda a: a[0], reverse=True)
        # print("after")
        # print(fps)
        nfs = []
        npp = []
        npf = []
        for item in fps:
            nfs.append(item[0])
            npp.append(item[1])
            npf.append(item[2])
            if len(nfs) >= len(self.popf):
                break
        self.pop = np.array(npp.copy())
        self.popf = np.array(npf.copy())
        self.fs = np.array(nfs.copy())
        self.pb = np.mean(self.popf, axis=0)
        self.covb = np.cov(self.popf.T)
        return self

    def predict0(self):
        p0 = self.pop
        p0f = self.popf
        # print(self.covb)
        # print(self.pb)
        # print("*")
        [p1f, p1] = self.get_pop()
        f0 = self.eval(p0)
        f1 = self.eval(p1)
        m10 = np.sign(np.sign(f1 - f0) + 1)
        m00 = 1 - m10  # avoid 0, 0
        m1 = m10[:, np.newaxis]
        m0 = m00[:, np.newaxis]
        self.pop = (p0 * m0) + (p1 * m1)
        self.popf = (p0f * m0) + (p1f * m1)
        self.fs = (f0 * m00) + (f1 * m10)
        self.pb = np.mean(self.popf, axis=0)
        self.covb = np.cov(self.popf.T)
        #for ind in self.popf:
        #    self.covb += (1/self.np)*(ind@ind.T)
        #self.covb += self.Q
        # print(self.pb)
        # print(self.covb)
        # print(f"Predict: self.pb->{self.pb}")
        # print(f"self.covb->{self.covb}")
        return self

    def update(self):
        eps = 0.00000000000001
        z = self.fs
        mu_z = np.mean(z)
        R = 0.
        Pz = np.var(z) + R
        x = self.popf
        mu_x = np.mean(x, axis=0)
        Pxz = np.zeros((self.nb, self.np))
        for i in x:
            i -= mu_x
            Pxz += i[:, np.newaxis]@(z - mu_z)[np.newaxis, :]
        Pxz /= (self.np * self.nb)
        #print(f"Pxz ->{Pz}, {np.shape(Pz)}")
        if abs(Pz) == 0.:
            Pz = 1
        K = Pxz/Pz
        # print(f"K ->{K}, {np.shape(K)}")
        y = z - mu_z
        Ky = K@y[:, np.newaxis]
        self.dpb = Ky[:, 0]
        self.dcb = Pz*(K@K.T)
        self.pb += self.dpb
        self.covb -= self.dcb
        return self

    def update0(self):
        meano = np.max(self.fs)
        yy = meano - self.fs
        std = np.std(yy)
        ws = np.ones(np.shape(yy))
        if std != 0:
            ws = norm.pdf(yy, loc=0., scale=std)
        covos = np.zeros(np.shape(self.covb))
        popa = np.copy(self.popf)
        popa *= ws[:, np.newaxis]
        for i in range(0, self.nb):
            for j in range(0, self.nb):
                covos[i, j] = popa[:, i]@popa[:, j].T
        self.covb = covos
        ws = ws[:, np.newaxis]
        self.pb = np.sum(ws * self.popf, axis=0) / sum(ws)
        return self

    def eval(self, pop):
        mapped = map(self.func.eval, pop)
        return np.asarray(list(mapped))


def main(argv):
    n = 0
    b = 0
    p = 0
    my_func = None
    try:
        opts, args = gt.getopt(argv, 'hn:b:p:f:')
    except gt.GetoptError:
        print('gba.py -h -n [1:inf] -b [1:inf] -p [1:inf] -f function name')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'gba.py -h -n [1:inf] -b [1:inf] -p [1:inf] -f function name')
            sys.exit(0)
        if opt == '-n':
            n = int(arg)
            continue
        if opt == '-b':
            b = int(arg)
            continue
        if opt == '-p':
            p = int(arg)
            continue
        if opt == '-f':
            if arg not in tf.func_dict:
                print('Unknown function.')
                sys.exit(2)
            else:
                my_func = tf.func_dict[arg]

        else:
            print(' Argument(s) not valid.')
            sys.exit(2)
    the_func = FObj(my_func, 0.8, 1, int(b/2), [0.8, 0.8], 'binary')
    bga = BGA(b, p, the_func)
    bga_dat = []
    print('inti pop')
    print(bga.pop)
    print('binary pop is')
    print(bga.popf)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(n):
        print('Generation: ', i, '/', n)
        f = bga.eval(bga.pop)
        bga_dat.append([i, np.mean(f), np.std(f), np.max(f)])
        # print('current dist: ', bga.pb)
        # print('current cov: ')
        # print(bga.covb)
        # print('pop, popf')
        # print(bga.pop)
        # print(bga.popf)
        # print(bga.pop.shape, bga.popf.shape)
        # print(f"mean before predict= {bga.pb}")
        # print(f"cov before predict =\n {bga.covb}")
        # print("p")
        bga.predict()
        # print(f"mean after predict= {bga.pb}")
        # print(f"cov after predict =\n {bga.covb}")
        # print("u")
        bga.update()
        # print('--------------------------------------------------------------')
    # print('Last update, ', bga.pb)
    pop = bga.pop
    f = bga.eval(pop)
    print('final pop')
    for ind, ff in zip(pop, f):
        print(f"{ind}->{ff}")
    print('fitness = ', np.mean(f), 'std = ', np.std(f), 'max = ', np.max(f))
    ind = []
    fmax = np.max(f)
    for i in range(0, bga.np):
        if abs(f[i] - fmax) < 0.00001:
            ind = pop[i]
            break
    print(f"ind is {ind}, thefuncm is {the_func.m}")
    indf = tf.pre_process(
        ind, [0, 1], int(bga.nb/the_func.m), modo='binary')
    print(f"@ {indf} -> {bga.func.eval(ind)}")
    my_f = FObj(bga.func.func, 0.8, 1, bga.func.m, [0.8, 0.8], 'real')
    print(f"@ {indf} -> {bga.func.eval(ind)}, {my_f.eval(indf)}")
    print("BGA")
    solo = [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    indf = tf.pre_process(
        solo, [0, 1], int(bga.nb/the_func.m), modo='binary')
    print(f"@ {solo} -> {bga.func.eval(solo)}, {indf}")
    print("BOA")
    solo = [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
    indf = tf.pre_process(
        solo, [0, 1], int(bga.nb/the_func.m), modo='binary')
    print(f"@ {solo} -> {bga.func.eval(solo)}, {indf}")
    # for po, fa in zip(pop, f):
    #    if fa >= mf:
    #        print(po)
    print('Func: ', my_func)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    bga_dat = np.vstack(bga_dat)
    plt.plot(bga_dat[:, 0], bga_dat[:, 1], color='g')
    plt.plot(bga_dat[:, 0], bga_dat[:, 1] + bga_dat[:, 2], 'g--')
    plt.plot(bga_dat[:, 0], bga_dat[:, 1] - bga_dat[:, 2], 'g--')
    plt.show()
#[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]->[0.815686274509803, 0.815686274509803]
#[1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 0]->[0.807843137254902, 0.807843137254902]


if __name__ == '__main__':
    main(sys.argv[1:])
