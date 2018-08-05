#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

import os

class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        #self.enc, self.dec = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)


    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=2):
    #def loss_enc(self, enc, x_out, t_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        #loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        if 0:
            x = np.array([[[-1, 0, 1, 299], [200, 0, 1, -1]],\
                          [[-1, 10, 1, 2], [20, 0, 11, -1]]]).astype(np.float32)
            t = np.array([[3, 0],[1,2]]).astype(np.int32)
            #x = np.reshape(x,(4,4))
            #t = np.reshape(t,(4))
            print(x,t)
            y = F.softmax_cross_entropy(x, t)
            print(y)
            #print(x_out, t_out)
            #x_out = x_out[0]
            #t_out = t_out[0][1]
            #x_out = np.transpose(x_out,[1,2,0])
            #t_out = np.transpose(t_out, [1, 2, 0])
            #print(x_out.shape, t_out.shape)
            #x_out = np.reshape(x_out,(65536,-1))
            #t_out = F.reshape(t_out, (65536,-1))
            #print(x_out.shape,t_out.shape)
            #y = F.softmax_cross_entropy(x_out, t_out)
            #print((np.reshape(np.transpose(x_out[0],[1,2,0]),(2,256*256))).shape)
            #print((np.reshape(t_out[0][1],(256*256))).shape)
            os.system('pause')
        loss_rec = lam1 * (F.sigmoid_cross_entropy(x_out, t_out))
        #loss_rec = lam1 * (F.softmax_cross_entropy(np.transpose(x_out[0],[1,2,0]), t_out[0][1]))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        #loss = loss_rec
        chainer.report({'loss': loss_rec}, enc)
        return loss

    #def loss_dec(self, dec, x_out, t_out, lam1=100, lam2=1):
    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=2):
        batchsize,_,w,h = y_out.data.shape
        #loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_rec = lam1 * (F.sigmoid_cross_entropy(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        #loss = loss_rec
        chainer.report({'loss': loss_adv}, dec)
        return loss
        
        
    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        #enc, dec = self.enc, self.dec
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in = 256
        w_out = 256
        
        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("i")
        t_out_f = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")
        
        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
            t_out_f[i, :] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        
        z = enc(x_in)
        x_out = dec(z)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out_f)


        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        #enc_optimizer.update(self.loss_enc, enc, x_out, t_out)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        #dec_optimizer.update(self.loss_dec, dec, x_out, t_out)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)
