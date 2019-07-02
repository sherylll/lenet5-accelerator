import matplotlib.pyplot as plt 
saved_graph_dir = 'keras_lenet_infer.png'

# CPU (Intel i7-7500 CPU @ 2.0GHz)
N = [10, 100, 1000, 10000]
cpu = [0.0078, 0.00047, 0.000219, 0.000199] # acceleration flattens out due to limited memory on a mobile cpu
# GPU (GeForce 940MX)
gpu = [0.2383,0.0128, 0.00132, 0.0002]

plt.plot(N, cpu, N, gpu)
plt.xscale('log')
plt.legend(['CPU','GPU'])
# plt.show()
plt.savefig(saved_graph_dir)
