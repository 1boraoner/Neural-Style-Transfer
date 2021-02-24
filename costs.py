import tensorflow as tf
import numpy as np
from gram import*
def cont_lost(cont,generated):
    x,nh,nw,nc = cont.shape
    cont_loss = (1/(4*nw*nh*nc))*tf.reduce_sum(tf.subtract(cont,generated)).numpy()
    return cont_loss

def style_lost_layer(style,generated):

    x,nh,nw,nc = style.shape
    style_unroll = style.reshape((x,nh*nw,nc))
    generated_unroll = generated.reshape((x,nh*nw,nc))

    Sgram = gram(style_unroll)
    Ggram = gram(generated_unroll)
    Diff = (Sgram - Ggram)
    jcost =  0.25 *(1/(nc*nh*nw)**2) * tf.reduce_sum(tf.square(Diff)).numpy()
    return jcost

def style_lost_total(style,generated,weight):
    jcost = 0
    for style_layer,generated_layer in zip(style,generated):
        jcost += weight*style_lost_layer(style_layer,generated_layer)
    return jcost


def cost(cont,style,gen, gen_style, weights, alpha,beta):

    J_cont = cont_lost(cont, gen)
    J_style = style_lost_total(style, gen_style,weights)
    J_total = alpha*J_cont + beta*J_style

    print(type(J_total))
    return J_total
