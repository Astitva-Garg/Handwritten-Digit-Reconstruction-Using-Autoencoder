import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main() -> None:
    transform = transforms.ToTensor()

    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)
    
    model = Autoencoder()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)

    epochs = 13
    outputs = []
    for epoch in range(epochs):
        for images, _ in data_loader:
            reconstructed = model(images)
            loss = mse_loss(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        if (epoch+1) % 1 == 0:  # every epoch
            outputs.append((epoch, images, reconstructed))
    
    for k in range(0, epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])
            
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1) # row_length + i + 1
            plt.imshow(item[0])

    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16 , 3, stride=2, padding=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(16, 32 , 3, stride=2, padding=1), 
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 64 , 7), 
                                     )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, 7),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid()
                                     )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == "__main__":
    main()
    
