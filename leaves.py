from neural_network import neuralNetwork
import os
from PIL import Image
import numpy


if __name__ == '__main__':
    input_nodes = 202500  # 28x28
    hidden_nodes = 1000
    output_nodes = 2
    learning_rate = 0.3

    count = 0
    n = neuralNetwork(inputnodes=input_nodes,hiddennodes=hidden_nodes,outputnodes=output_nodes,learningrate=learning_rate)


    files_0 = os.listdir('image_recognition/dataset/work2/0')
    for file in files_0:

        path = 'image_recognition/dataset/work2/0/'+file
        sImg = Image.open(path)
        w, h = sImg.size

        dImg = sImg.resize((int(w / 10), int(h / 10)), Image.ANTIALIAS)
        dImg = dImg.convert('L')
        image_arr = numpy.array(dImg)
        for i in range(image_arr.shape[0]):  # 转化为二值矩阵
            for j in range(image_arr.shape[1]):
                if image_arr[i, j] != 0:
                    image_arr[i, j] = 1
                else:
                    image_arr[i, j] = 0

        inputs = ((numpy.ravel(image_arr)) / 255.0 * 0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[0] = 0.99
        n.train(inputs, targets)
        print('训练完成'+str(count))

    files_1 = os.listdir('image_recognition/dataset/work2/1')
    for file in files_1:

        path = 'image_recognition/dataset/work2/1/' + file
        sImg = Image.open(path)
        w, h = sImg.size

        dImg = sImg.resize((int(w / 10), int(h / 10)), Image.ANTIALIAS)
        dImg = dImg.convert('L')
        image_arr = numpy.array(dImg)
        for i in range(image_arr.shape[0]):  # 转化为二值矩阵
            for j in range(image_arr.shape[1]):
                if image_arr[i, j] != 0:
                    image_arr[i, j] = 1
                else:
                    image_arr[i, j] = 0

        inputs = ((numpy.ravel(image_arr)) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[1] = 0.99
        n.train(inputs, targets)
        print('训练完成' + str(count))

    print('训练全部完成')

    for file in files_1:
        input('>>>input any key')
        path = 'image_recognition/dataset/work2/1/' + file
        sImg = Image.open(path)
        w, h = sImg.size

        dImg = sImg.resize((int(w / 10), int(h / 10)), Image.ANTIALIAS)
        dImg = dImg.convert('L')
        image_arr = numpy.array(dImg)
        for i in range(image_arr.shape[0]):  # 转化为二值矩阵
            for j in range(image_arr.shape[1]):
                if image_arr[i, j] != 0:
                    image_arr[i, j] = 1
                else:
                    image_arr[i, j] = 0

        result = n.query(((numpy.ravel(image_arr)) / 255.0 * 0.99) + 0.01)
        print(result)























