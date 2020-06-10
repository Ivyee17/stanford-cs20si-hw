import tensorboard as tb
import tensorflow as tf

tf=tf.compat.v1
tf.disable_eager_execution()

# with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
a1=tf.constant(2)
a=tf.add(3,a1)
b=tf.multiply(3,2)
c=tf.pow(a,b)

g=tf.Graph()
with g.as_default():
    a2=tf.add(3,5)

with tf.Session(config=tf.ConfigProto()) as sess:
    bb=sess.run(c)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(bb)

g=tf.get_default_graph()
sess2=tf.Session(graph=g)
res=sess2.run(a)
print(res)