import matplotlib.pylab as plt



def draw(accuracies):

    training = accuracies[0]
    testing =  accuracies[1]
    training_around1_accuracy = accuracies[2]
    testing_around1_accuracy = accuracies[3]
    training_around2_accuracy = accuracies[4]
    testing_around2_accuracy = accuracies[5]

    ks =[ i for i in range(len(training))]
    f1 = plt.figure(1)
    plt.title('accuracy')
  #  plt.xticks(ks)
    plt.ylim([0, 1])
    plt.plot(ks,training, color='red',label='training',alpha=0.5)
  #  plt.plot(ks,validation, color='blue',label='validation',alpha=0.5)
    plt.plot(ks,testing, color='green',label='testing',alpha=0.5)

    plt.legend(loc='upper right')

    ks = [i for i in range(len(training))]
    f2 = plt.figure(2)
    plt.title('around 1 accuracy')
    #  plt.xticks(ks)
    plt.ylim([0, 1])
    plt.plot(ks, training_around1_accuracy, color='red', label='training_around1_accuracy', alpha=0.5)
    #  plt.plot(ks,validation, color='blue',label='validation',alpha=0.5)
    plt.plot(ks, testing_around1_accuracy, color='green', label='testing_around1_accuracy', alpha=0.5)

    plt.legend(loc='upper right')

    ks = [i for i in range(len(training))]
    f3 = plt.figure(3)
    plt.title('around 2 accuracy')
    #  plt.xticks(ks)
    plt.ylim([0, 1])
    plt.plot(ks, training_around2_accuracy, color='red', label='training_around2_accuracy', alpha=0.5)
    #  plt.plot(ks,validation, color='blue',label='validation',alpha=0.5)
    plt.plot(ks, testing_around2_accuracy, color='green', label='testing_around2_accuracy', alpha=0.5)

    plt.legend(loc='upper right')
    plt.show()