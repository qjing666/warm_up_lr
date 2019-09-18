from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import numbers
import paddle.fluid as fluid
from paddle.fluid.layers import control_flow
from paddle.fluid.layers import nn
from paddle.fluid.layers import ops
from paddle.fluid.layers import tensor
from paddle.fluid.layers import learning_rate_scheduler
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.framework import default_main_program, Parameter, unique_name, name_scope
from paddle.fluid.framework import Variable
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.dygraph import learning_rate_scheduler as imperate_lr
def _decay_step_counter(begin=0):
    # the first global step is zero in learning rate decay
    global_step = nn.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    global_step = tensor.cast(global_step, 'float32')
    return global_step

def lr_warmup(learning_rate, warmup_steps, total_step,multiplier,step_each_epoch):
    with default_main_program()._lr_schedule_guard():
        lr = tensor.create_global_var(shape=[1],value=0.0,dtype='float32',persistable=True,name='learning_rate_warmup')
        global_step = _decay_step_counter()

        with control_flow.Switch() as switch:
            with switch.case(global_step<=warmup_steps):
                decay_lr = learning_rate*((multiplier - 1.) * global_step / warmup_steps + 1.)
                tensor.assign(decay_lr,lr)
            with switch.default():
                learning_rate = learning_rate*multiplier
                #cur_epoch = ops.floor(global_step/step_each_epoch)
                decay_lr = learning_rate * 0.5 * (ops.cos((global_step-warmup_steps)*math.pi/(total_step))+1)
                tensor.assign(decay_lr,lr)

    return lr

if __name__ == '__main__':
    learning_rate = 0.01
    warmup_steps = 10
    total_step = 100
    multiplier = 8
    step_each_epoch = 1000
    x = []
    y1 = []
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    decayed_lr = lr_warmup(learning_rate,warmup_steps,total_step,multiplier,step_each_epoch)
    exe.run(program=fluid.default_startup_program())
    for i in range(total_step):
        x.append(i)
        lr = exe.run(program=fluid.default_main_program(), fetch_list=["learning_rate_warmup"])
        #print(lr[0][0])
        y1.append(lr[0][0])

y2 = []
with open('result') as newfile:
    for line in newfile.readlines():
        y2.append(float(line[:-1]))

l1 = plt.scatter(x, y1, color='blue',marker='x',label='paddle')
l2 = plt.scatter(x, y2, color='red',marker='+',label='torch')
plt.legend(handles=[l1, l2], labels=['paddle', 'torch'], loc='best')
plt.show()
