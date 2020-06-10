import tensorflow as tf
tf=tf.compat.v1
tf.disable_eager_execution()

a=tf.constant([1,2])
b=tf.constant([[3,4],[5,6]])
c=tf.fill(tf.zeros_like([[2]]).shape,6)
print(c)
x=tf.multiply(a,c)
with tf.Session() as sess:
    r=sess.run(x)
    print(r)
    tf.summary.FileWriter('./graphs',sess.graph)
