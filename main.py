import neural

INPUT_NODES = 2
HIDDEN_NODES = 10
OUTPUT_NODES = 1

LEARNING_RATE = 0.5

myNetwork = neural.neuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
# myNetwork.__str__()

# XOR:
# 0,0 = 0
# 0,1 = 1
# 1,0 = 1
# 1,1 = 0

n = 1
for i in range(100000):
    if (n == 1):
        myNetwork.train([0,0], [0])
        n += 1
    elif (n == 2):
        myNetwork.train([0,1], [1])
        n += 1
    elif (n == 3):
        myNetwork.train([1,0], [1])
        n += 1
    else:
        myNetwork.train([1,1], [0])
        n = 1

print(f'\n0 XOR 0 = {myNetwork.query([0, 0])}')
print(f'\n0 XOR 1 = {myNetwork.query([0, 1])}')
print(f'\n1 XOR 0 = {myNetwork.query([1, 0])}')
print(f'\n1 XOR 1 = {myNetwork.query([1, 1])}')