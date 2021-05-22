import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import numpy as np
""" short and sweet vae, < 100 lines guaranteed! """

train_dset = torchvision.datasets.MNIST('dataset/mnist/', download=True, train=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self, h1=512, h2=256, h3=128, z_dim=2):
        super(VAE, self) .__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(28*28, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, h3),
                    nn.ReLU(),
                    )
        self.head_mean = nn.Linear(h3, z_dim)
        self.head_var = nn.Linear(h3, z_dim)      # the output is log(var) NOT var
        self.decoder = nn.Sequential(
                    nn.Linear(z_dim, h3),
                    nn.ReLU(),
                    nn.Linear(h3, h2),
                    nn.ReLU(),
                    nn.Linear(h2, h1),
                    nn.ReLU(),
                    nn.Linear(h1, 28*28),
                    nn.Sigmoid(),
                    )

    def forward(self, X):
        return self.decode(*self.encode(X)) # autoencoders in a nutshell

    def encode(self, X):
        silence = self.encoder(X)
        return self.head_mean(silence), self.head_var(silence)

    def decode(self, mu, log_var):
        return self.decoder(self.sample(mu, log_var))

    def sample(self, mu, log_var):
        eps = torch.rand_like(mu).cuda()
        return mu + eps * torch.exp(log_var*.5)

def loss_func(X_hat, X, mu, log_var):
    rec_loss = torch.nn.functional.binary_cross_entropy(X_hat, X, reduction='sum')
    kl_loss = -.5 * torch.sum((1 + log_var - torch.square(mu) - log_var.exp()))
    return rec_loss + kl_loss
    
def train(model, epochs, cpt_dir='models/vae.pt'):

    model.train().cuda()
    for e in range(epochs):
        print("="*20 + " current epoch %d " % e + "="*20)
        for it, (X, _) in enumerate(train_loader):
            X = X.to(device='cuda').view(-1, 28*28)
            mu, log_var = model.encode(X)
            X_hat = model.decode(mu, log_var)
            loss = loss_func(X_hat, X, mu, log_var)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            print("="*20 + " iteration {} got loss {:.4f}".format(it, loss) + "="*20)

    torch.save(model.state_dict(), cpt_dir)
    print("model done training, saved to %s" % cpt_dir)

if __name__ == "__main__":

    vae = VAE()
    vae.load_state_dict(torch.load('models/vae.pt'))
    vae.cuda().eval()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    #train(vae, 1)

    images = train_dset[np.random.randint(len(train_dset))][0].to(device='cuda').view(-1, 1, 28, 28)
    with torch.no_grad():
        #x = torch.linspace(-1, 1, 16).to(device='cuda')
        #sample = vae.decoder(torch.dstack(torch.meshgrid(x, x)).view(-1, 2)).view(-1, 1, 28, 28)
        #noisy_img = (images[0] + torch.normal(0, .05, img.shape).to(device='cuda')).clamp(0, 1)
        for i in range(100):
            print(" {} almost there".format(i+1))
            rec_img = vae(images[-1].view(-1, 28*28)).view(-1, 1, 28, 28)
            rec_img = (rec_img + torch.normal(0, .05, rec_img.shape).to(device='cuda')).clamp(0, 1)
            images = torch.vstack((images, rec_img))

    #torchvision.utils.save_image(torch.vstack((noisy_img, rec_img)), link, nrow=16)
    dir_ = 'images/img_rec_loop2.jpeg'
    save_image(images, dir_, nrow=16)
    img = matplotlib.image.imread(dir_)
    plt.imshow(img)
    plt.show()
