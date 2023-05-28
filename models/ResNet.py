"""
Implementation of ResNet, from paper
"Deep Residual Learning for Image Recognition" by Kaiming He et al.

url: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
"""

import torch
import torch.nn as nn

def conv3x3(input, output, stride=1, groups=1, dilation=1):
    """
    3x3 convolution layer
    bias = False : BatchNorm에 bias가 포함되어있어, conv2d는 bias=False로 설정
    """
    return nn.Conv2d(input, output, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dialtion=dilation)

def conv1x1(input, output, stride=1):
    """1x1 convolutionlayer"""
    return nn.Conv2d(input, output, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, input, output, stride=1):
        super.__init__()
        #batchNorm에 bias가 포함되어 있으므로 bias=False로 설정
        self.residual_function = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output * BasicBlock.expansion, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(output * BasicBlock.expansion),
        )
        #input과 output의 feature map size, filter수가 동일한 idetity mapping사용
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        #using 1x1 conv to projection mapping

        if stride !=1 or input != BasicBlock.expasion * output:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input, output * BasicBlock.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(output * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
class BottleNeck(nn.Module):
    expansion=4 # 
    def __init__(self, input, output, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output, kernel_size=3, stride=stride, 
                      padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output * BottleNeck.expansion),
        )
        
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or input != output * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input, output*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10,
                 init_weights=True):
        super().__init__()

        self.input=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input=3, ouput=64, kernel_size=7, stride=2, 
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #weight inittialization
        if init_weights:
            self._initialinze_wieghts()

    def _make_layer(self, block, output, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input, output, stride))
            self.input = output * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    #define weight initailization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode= 'fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.noraml_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])