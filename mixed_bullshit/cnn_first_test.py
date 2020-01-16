import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#im = Image.open(name+".ppm")
#name = "data/TraficSigns/BackdoorImages/train/Original/Stop_G_00000_00002 (1)"

x_input = [[-10],  # <- Angstzustand Gast 1
           [-5],  # <- Angstzustand Gast 2
           [-2],  # <- Angstzustand Gast 3
           [-1],
           [2],
           [1],
           [6],
           [9]]   # <- Angstzustand Gast 8
# 
# gewünschtes Ausgangssignal, Tensor
# Endzustand der Gäste -> Wunsch die Bahn nie gefahren zu sein
y_input = [[0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [1],  # <- bereut die Fahrt
           [1]]  # <- bereut die Fahrt

wag = tf.compat.v1.placeholder(tf.float32, shape=[8, 1])

y_true = tf.compat.v1.placeholder(tf.float32, shape=[8, 1])

# Variable
# Geschwindigkeit des Wagons
v = tf.Variable([[1.0]])  # tf.random_normal(shape=[1,1], mean=2, stddev=1)
# Starthöhe des Wagons
h = tf.Variable([[-2.0]])  # tf.random_normal(shape=[1,1])
# Knoten mit Matrizenoperator, Fahrelement, z.B. Airtime-Hügel
z = tf.matmul(wag, v) + h

# Knoten mit ReLu-Aktivierungsfunktion
y_pred = tf.nn.relu(z)
# Fehlerfunktion
err = tf.square(y_true - y_pred)
# Optimierer
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(err)
# Initialisierung der Variablen (Geschwindigkeit)
init = tf.compat.v1.global_variables_initializer()

runden = 200
# Array zum notieren der Größen
v_array = []
h_array = []
# Aufzeichnung des Fehlerwertes
loss = []

# Ausführung des Graphen
with tf.compat.v1.Session() as sess:
    # Initialisierung dar Variablen
    sess.run(init)
    # Beginn der 100 Fahrten
    for i in range(runden):
        # Ausgabe der Werte
        _, geschw, hoehe, Y_pred, error = sess.run([opt, v, h, y_pred, err],
                                                   feed_dict={wag: x_input,
                                                              y_true: y_input
                                                              }
                                                   )
        loss.append(np.mean(error))
        v_array.append(float(geschw))
        h_array.append(float(hoehe))
# Ausgabe der Werte der letzten Runde
print('Angstlvl berechnet: \n{}'.format(Y_pred))
print('\n Geschwindigkeit: \n{}'.format(geschw))
print('\n Starthöhe: \n{}'.format(hoehe))
print('\n Fehler: \n{}'.format(error))


with plt.xkcd():
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].plot(range(runden), loss)
    ax[0].set_xlabel('Runden/Epochen')
    ax[0].set_ylabel('$err$')
    ax[0].title.set_text('Fehlerverlauf')
    ax[1].plot(range(runden), v_array, label='Geschwindigkeit')
    ax[1].plot(range(runden), h_array, label='Bias')
    ax[1].set_xlabel('Runden/Epochen')
    ax[1].set_ylabel('$v, h$')
    ax[1].legend()
    ax[1].title.set_text('Geschwindigkeits- und Höhenverlauf')
plt.savefig('achterbahn_result.png')
