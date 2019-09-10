import numpy as np
import matplotlib.pyplot as plt
import pickle

# ========== EXTRACCIÓN DE DATOS ==============
archivo = open('/home/endredra/Desktop/Int_proj/EL4106-1oProyecto/dataset/HiTS2013_100k_samples(4_channels)_images_labels.pkl',"rb")
example_dict = pickle.load(archivo)

print()
print(example_dict['images'].shape)

print()
xx = example_dict['images'][:, :, :, 1]
print(xx[0].shape)

#x = np.zeros((2, 3, 4, 4))
#print(x.shape)
#print()
#print(x)
#print()

# ========== ARREGLO DE EJEMPLO ===============
#x = np.arange(20)
#x = x.reshape(4, 5)
#print('Arreglo de prueba :')
#print(x)
#print()

#print(x.shape)
#print(x.shape[1])
#print()

x = xx[23]
plt.imshow(x)
plt.show()
print()

# ========== ESTADISTICAS ====================
print('       ESTADISTICAS      ')
print('Valor máximo :', x.max())
print('Valor mínimo :', x.min())
print('Valor promedio :', x.mean())
print('Valor varianza :', x.var())
print('Valor desviación standar :', x.std())
