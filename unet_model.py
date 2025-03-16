import torch
import torch.nn as nn
import torch.nn.functional as F 

#Double Convolutional Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# U-Net encoder
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            in_channels = feature  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsampling layer

    def forward(self, x):
        skip_connections = []  # store feature maps for skip connections
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)  # downsample
        return x, skip_connections  # f.maps for skip connections

# U-Net Decoder 
class UNetDecoder(nn.Module):
    def __init__(self, out_channels, features=[512, 256, 128, 64]):
        super(UNetDecoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(len(features)):
            if i == 0:
                self.upconvs.append(nn.ConvTranspose2d(features[i], features[i], kernel_size=2, stride=2))
            else:
                self.upconvs.append(nn.ConvTranspose2d(features[i] * 2, features[i], kernel_size=2, stride=2))

            self.dec_blocks.append(DoubleConv(features[i] * 2, features[i]))

        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)  # 1x1 conv for final segmentation

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]  #reverse order to match decoder 

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)  # upsample
            skip = skip_connections[i]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip, x), dim=1)  # concatenate with encoder f.map
            x = self.dec_blocks[i](x)  # apply conv. block

        return self.final_conv(x)  # output segmentation mask

# full U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)  # encoder outputs features and skip connections
        output = self.decoder(bottleneck, skip_connections)  # decoder reconstructs segmentation
        return output


if __name__ == "__main__":
    test_input = torch.randn(1, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=4)  # 4 segmentation classes
    output = model(test_input)

    print("U-Net output shape:", output.shape)


