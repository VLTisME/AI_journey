from utils import *

class ResidualBlock(Model, Layer): # Layer?
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, (3, 3), strides = stride, padding = 'same', use_bias = False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channels, (3, 3), strides = 1, padding = 'same', use_bias = False)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(out_channels, (1, 1), strides = stride),
                layers.BatchNormalization()
            ])  
            
        self.relu = layers.ReLU()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            inputs = self.downsample(inputs)
        x += inputs
        return self.relu(x)

class ResNet_18(Model):
    def __init__(self, residual_block, n_blocks_list, nclasses):
        super(ResNet_18, self).__init__()
        
        # Initial blocks
        self.conv1 = layers.Conv2D(64, (3, 3), strides = 1, padding = 'same', use_bias = False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        # Residual blocks
        self.block1 = self.create_layer(residual_block, 64, 64, n_blocks_list[0], stride = 1)
        self.block2 = self.create_layer(residual_block, 64, 128, n_blocks_list[1], stride = 2)
        self.block3 = self.create_layer(residual_block, 128, 256, n_blocks_list[2], stride = 2)
        self.block4 = self.create_layer(residual_block, 256, 512, n_blocks_list[3], stride = 2)

        # Fully connected layers
        self.global_avg_pooling = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(nclasses)
    
    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        layers_list = []
        # first block may contain stride > 1
        layers_list.append(residual_block(in_channels, out_channels, stride))
        # other blocks
        for _ in range(n_blocks):
            layers_list.append(residual_block(out_channels, out_channels, 1))
        return tf.keras.Sequential(layers_list)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) # when fitting, high-level api keras automatically detect whether it is training or inferencing so it automatically set 'training = True/False'
        x = self.global_avg_pooling(x)
        x = self.fc(x)

        return x

n_blocks_list = [2, 2, 2, 2]
n_classes = 10
model = ResNet_18(ResidualBlock, n_blocks_list, n_classes)

# BN is applied per channel in conv layer-> each channel has gamma and beta
# BN is applied per feature of all datapoints from the minibatch -> imagine it's like a NN, only one NN... imagine it's like cai plot 2D toa do cac diem la (x1, x2), thi neu take only x1 thi ta can lay moi x1 - u roi chia std cua mot thang x1 thoi de no co mean = 0, std = 1, di chuyen cac diem do xung quanh toa do (0, 0)
