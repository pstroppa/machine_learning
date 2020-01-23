epoch 50: poisonous 9 Classes GoStraightandTurn
test loss, testacc:  [13.778729133605957, 0.05999999865889549]

epoch 100: poisonous 9 Classes GoStraightandTurn
test loss, testacc:  [21.23149169921875, 0.019999999552965164]


TARGTED ATTACK ON CAN-GO-STRAIGHT-AND-TURN



value tweaking:

DROPP_Rate_paa = 0.995          0.998           0.998
n_epochs_ paa = 100             100             100
learning_rate_paa = 0.0007      0.0007          0.0007
train_test_ratio_paa = 0.05     0.05            0.05
bias_decrease = 0.4             0.3             0.3

result: cl:0.968, poi:0.639 | cl:0.980, poi:0,54 |  cl:0.979, poi:0.56

