import tensorflow as tf

# x = tf.placeholder("float",None)
# y = x**2

# with tf.Session() as session:
# 	result = session.run(y,feed_dict = {x:[1,2,3]})
# 	print(result)

# a = tf.constant([2])
# b = tf.constant([3])

a = tf.constant(2)
b = tf.constant(3)

c = a + b
d = tf.add(a,b)

session = tf.Session()
result = session.run(c)
result1 = session.run(d)

print(result,end = " ")
print(result1)