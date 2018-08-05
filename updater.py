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
        self.enc, self.dec, self.dis, self.enc_removal, self.dec_removal, self.dis_removal = kwargs.pop('models')
        #self.enc, self.dec = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)


    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=2):
    #def loss_enc(self, enc, x_out, t_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1 * (F.sigmoid_cross_entropy(x_out, t_out))
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

    def loss_enc_removal(self, enc_removal, x_out_removal, t_out_removal, y_out_removal, lam3 = 1, lam4 = 1):
        batchsize, _, w, h = y_out_removal.data.shape
        loss_rec = lam3 * F.mean_absolute_error(x_out_removal, t_out_removal)
        loss_adv = lam4 * F.sum(F.softplus(-y_out_removal)) / batchsize / w / h
        loss = loss_rec + loss_adv
        # chainer.report({'loss': loss}, dis)
        return loss

    def loss_dec_removal(self, des_removal, x_out_removal, t_out_removal, y_out_removal, lam3 = 1, lam4 = 1):
        batchsize, _, w, h = y_out_removal.data.shape
        loss_rec = lam3 * F.mean_absolute_error(x_out_removal, t_out_removal)
        loss_adv = lam4 * F.sum(F.softplus(-y_out_removal)) / batchsize / w / h
        loss = loss_rec + loss_adv
        # chainer.report({'loss': loss}, dis)
        return loss

    def loss_dis_removal(self, dis_removal, y_in_removal, y_out_removal):
        batchsize, _, w, h = y_in_removal.data.shape
        L1 = F.sum(F.softplus(-y_in_removal)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out_removal)) / batchsize / w / h
        loss = L1 + L2
        #chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')

        enc_removal_optimizer = self.get_optimizer('enc_removal')
        dec_removal_optimizer = self.get_optimizer('dec_removal')
        dis_removal_optimizer = self.get_optimizer('dis_removal')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        enc_removal, dec_removal, dis_removal = self.enc_removal, self.dec_removal, self.dis_removal

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
        t_out_removal = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        
        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
            t_out_f[i, :] = xp.asarray(batch[i][1])
            t_out_removal[i, :] = xp.asarray(batch[i][2])
        x_in = Variable(x_in)
        
        z = enc(x_in)
        x_out = dec(z)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out_f)


        z_removal = enc_removal(x_in, x_out)
        x_out_removal = dec_removal(z_removal)

        y_removal_fake = dis_removal(x_in,x_out,x_out_removal)
        y_removal_real = dis_removal(x_in,t_out_f,t_out_removal)



        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        #enc_optimizer.update(self.loss_enc, enc, x_out, t_out)
        #for z_ in z:
            #z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        #dec_optimizer.update(self.loss_dec, dec, x_out, t_out)
        #x_in.unchain_backward()
        #x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)

        enc_removal_optimizer.update(self.loss_enc_removal, enc_removal, x_out_removal, t_out_removal, y_removal_fake)
        dec_removal_optimizer.update(self.loss_dec_removal, dec_removal, x_out_removal, t_out_removal, y_removal_fake)
        dis_removal_optimizer.update(self.loss_dis_removal, dis_removal, y_removal_real, y_removal_fake)
