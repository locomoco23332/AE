import os

# from turtle import forward
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
# print(current_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class LN(nn.Module):
    def __init__(self,dim,epsilon=1e-5):
        super().__init__()
        self.eps=epsilon

        self.alpha=nn.Parameter(torch.ones([1,dim,1]),requires_grad=True)
        self.beta=nn.Parameter(torch.zeros([1,dim,1]),requires_grad=True)

    def forward(self,x):
        mean=x.mean(axis=1,keepdim=True)
        var=((x-mean)**2).mean(dim=1,keepdim=True)
        std=(var+self.eps).sqrt()
        y=(x-mean)/std
        y=y*self.alpha+self.beta
        return y



class NormalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    #        self.fc5 = nn.Linear(hidden_size,hidden_size)
    #        self.fc6 = nn.Linear(hidden_size,output_size)

    def forward(self, input_data):
        input_data = input_data.flatten(-2)
        out = self.fc1(F.elu(input_data))
        out = self.fc2(F.elu(out))
        out = self.fc3(F.elu(out))
        #        out4 = self.fc4(F.elu(out3))
        #        out5 = self.fc4(F.elu(out4))
        return self.fc4(out)

    def set_normalization(self, mu, std):
        self.mu = mu
        self.std = std

    def normalize(self, t):
        return (t - self.mu) / self.std

    def denormalize(self, t):
        return t * self.std + self.mu


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size,num_condition):
        super().__init__()
        self.input_size = input_size*num_condition
        self.latent_size = latent_size
        self.hidden_size = 54
        real_input = self.input_size
        self.fc1 = nn.Linear(real_input, self.hidden_size)
        self.fc2 = nn.Linear( self.hidden_size*num_condition, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size*num_condition, latent_size)
        self.var = nn.Linear(self.hidden_size*num_condition, latent_size)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input, condition_input):
        out1 = F.elu(self.fc1(torch.cat((input, condition_input), dim=1)))

        out2 = F.elu(self.fc2(torch.cat((input, out1), dim=1)))
        out3 = torch.cat((input, out2), dim=1)
        return self.mu(out3), self.var(out3)

    def forward(self, input, condition_input):
        mu, var = self.encode(input, condition_input)
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    def __init__(self, input_size, latent_size, num_experts, output_size):
        super().__init__()
        self.input_size = latent_size + input_size
        output_size = output_size*num_experts
        hidden_size = 256
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size , hidden_size)
        self.fc3 = nn.Linear( hidden_size, output_size)

    def forward(self, z, condition_input):
        # print("latent_shape : ",z.shape)
        out4 = F.elu(self.fc1(torch.cat((z, condition_input), dim=1)))
        #out5 = F.elu(self.fc2(torch.cat((z, out4), dim=1)))
        out5=F.elu(self.fc2(out4))
        return self.fc3(out5)
        #return self.fc3(torch.cat((z, out5), dim=1))


class MixedDecoder(nn.Module):
    def __init__(
            self,
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
            num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z):
        coefficients = F.softmax(self.gate(z), dim=1)
        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = z.unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class VAE(nn.Module):
    def __init__(self, input_size, latent_size,num_condition, num_experts, output_size):
        super().__init__()
        self.encoder = Encoder(input_size, latent_size,num_condition)
        # self.decoder = MixedDecoder(input_size,latent_size,256,1,1,num_experts)
        self.decoder = Decoder(input_size, latent_size, num_experts, output_size)

        ############################change initialization orer##########################3
        self.data_std = 0
        self.data_avg = 0
        ############################change initialization orer##########################3
        self.latent_list = []

    def encode(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return z, mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c):
        return self.decoder(z, c)

    def set_normalization(self, std, avg):
        self.data_std = std
        self.data_avg = avg

    def set_latent_list(self, latent_vectors):
        self.latent_list = latent_vectors

    #######################
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std

    def denormalize(self, t):
        return t * self.data_std + self.data_avg
    #######################


class BetaDerivatives():
    def __init__(self, time_steps, beta_start, beta_end):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        self.betas = self.prepare_noise_schedule().to(device="cpu")
        self.alpha = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.time_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.time_steps - 1, size=(n,))

    def gather(self, a, t):
        return torch.gather(a, 1, t)


class GaussianDiffusion():
    def __init__(self, input_size, noise_step, output_size):
        self.device = "cpu"
        self.input_size = input_size
        self.output_size = output_size
        self.noise_step = noise_step
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betaderivative = BetaDerivatives(noise_step, self.beta_start, self.beta_end)

        self.beta = self.betaderivative.prepare_noise_schedule().to(self.device)
        self.alpha = self.betaderivative.alpha
        self.alpha_hat = self.betaderivative.alpha_hat

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn((t.shape[0], x_0.shape[0]))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise.to(self.device)


class TimeEmbedding(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)

    def activation(self, x):
        return x * F.sigmoid(x)

    def forward(self, t):
        half_dim = self.n // 2
        emb = torch.log(torch.tensor(1000.0) / (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.activation(self.fc1(emb))
        emb = self.fc2(emb)
        return emb


class DenoiseDiffusion(nn.Module):
    def __init__(self, input_size, output_size, noise_steps):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.noise_steps = noise_steps
        self.hidden_size = 32
        self.time_dim = self.hidden_size
        self.gaussiandiffusion = GaussianDiffusion(self.input_size, self.noise_steps, self.output_size)
        self.timeembedding = TimeEmbedding(self.time_dim)
        self.betas = self.gaussiandiffusion.beta
        self.alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.time_dim, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.time_dim, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.time_dim, self.output_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt, t):
        eps_theta = self.forward(xt, t)
        alpha_hat = self.gaussiandiffusion.alpha_hat[t]
        alpha = self.gaussiandiffusion.alpha[t]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[t]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, xt, t):
        t = self.timeembedding(t)
        emb = self.fc1(xt)
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc2(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc3(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc4(emb))
        return emb


class DanceEncoder10(nn.Module):
    def __init__(self, pose_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = pose_size * 10
        self.pose_size = pose_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, self.latent_size)
        self.std = nn.Linear(self.hidden_size, self.latent_size)

    def encode(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        data = torch.cat((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(out1))
        out3 = self.fc3(F.elu(out2))
        out4 = self.fc4(F.elu(out3))
        return self.mu(out4), self.std(out4)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        mu, var = self.encode(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        z = self.reparameterize(mu, var)
        return z, mu, var


class DanceDecoder10(nn.Module):
    def __init__(self, latent_size, pose_size, hidden_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        self.pose_size = pose_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.pose_size * 5 + self.latent_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.pose_size * 5, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.pose_size * 5, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.pose_size * 5, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + self.pose_size * 5, self.output_size)

    def forward(self, z, t1, t2, t3, t4, t5):
        out1 = self.fc1(F.elu(torch.cat((z, t1, t2, t3, t4, t5), dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1, t1, t2, t3, t4, t5), dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2, t1, t2, t3, t4, t5), dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3, t1, t2, t3, t4, t5), dim=1)))
        return self.fc5(torch.cat((out4, t1, t2, t3, t4, t5), dim=1))


class DanceVAE10(nn.Module):
    def __init__(self, pose_size, encode_hidden_size, latent_size, decode_hidden_size, output_size):
        super().__init__()
        self.encoder = DanceEncoder10(pose_size, encode_hidden_size, latent_size)
        self.decoder = DanceDecoder10(latent_size, pose_size, decode_hidden_size, output_size)
        self.pose_data_mu = 0
        self.pose_data_std = 0

    def encode(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        z, mu, logvar = self.encoder(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        return z, mu, logvar

    def forward(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        z, mu, logvar = self.encoder(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        return self.decoder(z, t1, t3, t5, t7, t9), mu, logvar

    def sample(self, z, t1, t3, t5, t7, t9):
        return self.decoder(z, t1, t3, t5, t7, t9)

    def set_normalize(self, pose_mu, pose_std):
        self.pose_data_mu = pose_mu
        self.pose_data_std = pose_std

    def normalize_pose(self, x):
        return (x - self.pose_data_mu) / self.pose_data_std

    def denormalize_pose(self, x):
        return x * self.pose_data_std + self.pose_data_mu


class TrackerEncoder(nn.Module):
    def __init__(self, tracker_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = tracker_size*10
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size + self.input_size, self.latent_size)
        self.std = nn.Linear(self.hidden_size + self.input_size, self.latent_size)

    def encode(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        data = torch.cat((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(torch.cat((out1, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)))
        return self.mu(torch.cat((out4, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1)), self.std(
            torch.cat((out4, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), dim=1))

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        mu, var = self.encode(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        z = self.reparameterize(mu, var)
        return z, mu, var


class TrackerDecoder(nn.Module):
    def __init__(self, latent_size, tracker_size, hidden_size, output_size):
        super().__init__()
        self.tracker_size = tracker_size
        self.input_size = latent_size + self.tracker_size * 5
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + latent_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + latent_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + latent_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + latent_size, self.output_size)

    def forward(self, z, t1, t3, t5, t7, t9):
        out1 = self.fc1(F.elu(torch.cat((z, t1, t3, t5, t7, t9), dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1, z), dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2, z), dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3, z), dim=1)))
        return self.fc5(torch.cat((out4, z), dim=1))


class TrackerVAE(nn.Module):
    def __init__(self, tracker_size, encode_hidden_size, latent_size, decode_hidden_size, output_size):
        super().__init__()
        self.encoder = TrackerEncoder(tracker_size, encode_hidden_size, latent_size)
        self.decoder = TrackerDecoder(latent_size, tracker_size, decode_hidden_size, output_size)

    def encode(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        z, mu, logvar = self.encoder(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        return z, mu, logvar

    def forward(self, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        z, mu, logvar = self.encoder(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
        return self.decoder(z, t1, t3, t5, t7, t9), mu, logvar

    def sample(self, z, t1, t3, t5, t7, t9):
        return self.decoder(z, t1, t3, t5, t7, t9)


class TrackerAutoEncoder(nn.Module):
    def __init__(self, tracker_size, num_condition_frames, hidden_size, latent_size):
        super().__init__()
        self.input_size = tracker_size * num_condition_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.latent_size)

    def encode(self, tracker_data):
        data = tracker_data.flatten(-2)
        out = F.elu(self.fc1(data))
        out = F.elu(self.fc2(torch.cat((out, data), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, data), dim=1)))
        out = self.fc4((torch.cat((out, data), dim=1)))
        return out

    def forward(self, tracker_data):
        latent = self.encode(tracker_data)
        return latent


class TrackerAutoDecoder(nn.Module):
    def __init__(self, latent_size, tracker_size, num_condition_frames, hidden_size, output_size):
        super().__init__()
        self.input_size = latent_size
        self.tracker_size = tracker_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.output_size)

    def forward(self, latent, tracker):
        out = F.elu(self.fc1(latent))
        out = F.elu(self.fc2(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, latent), dim=1)))
        out = self.fc4(torch.cat((out, latent), dim=1))
        return out


class TrackerAuto(nn.Module):
    def __init__(self, tracker_size, num_condition_frames, encoder_hidden_size, latent_size, decoder_hidden_size,
                 output_size):
        super().__init__()
        self.encoder = TrackerAutoEncoder(tracker_size, num_condition_frames, encoder_hidden_size, latent_size)
        self.decoder = TrackerAutoDecoder(latent_size, tracker_size, num_condition_frames, decoder_hidden_size,
                                          output_size)
        self.num_condition_frames = num_condition_frames
        # self.decoder = MixedDecoder(35,latent_size,decoder_hidden_size,0,1,2)

    def forward(self, tracker_data):
        z = self.encoder(tracker_data)
        return self.decoder(z, tracker_data[:, int(self.num_condition_frames / 2 - 1), :])


class CNN(nn.Module):
    def __init__(self, tracker_size, condition_size, output_size):
        super().__init__()
        self.tracker_size = tracker_size * condition_size
        self.output_size = output_size
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 512, (1, 3), stride=(1, 3)),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 32, 1),
            torch.nn.ELU(),
        )
        self.fc1 = torch.nn.Linear(5760, 1024)
        self.fc2 = torch.nn.Linear(1024, output_size)

    def forward(self, history):
        history = history.unsqueeze(1)
        # print(history.shape)
        out = self.layer1(history)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        out = F.elu(self.fc1(out))
        out = self.fc2(out)
        # print(out.shape)
        return out
class PoseEncoder(nn.Module):
    def __init__(self,
                 frame_size,
                 latent_size,
                 hidden_size,
                 num_condition_frames,
                 num_future_predictions,):
        super().__init__()
        input_size=frame_size*2
        self.fc1=nn.Linear(input_size,hidden_size)
        self.fc2=nn.Linear(frame_size+hidden_size,hidden_size)
        self.mu=nn.Linear(frame_size+hidden_size,latent_size)
        self.logvar=nn.Linear(frame_size+hidden_size,latent_size)
    def encode(self,x,c):
        h1=F.elu(self.fc1(torch.cat((x,c),dim=1)))
        h2=F.elu(self.fc2(torch.cat((x,h1),dim=1)))
        s=torch.cat((x,h2),dim=1)
        return self.mu(s),self.logvar(s)
    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    def forward(self,x,c):
        mu,logvar=self.encode(x,c)
        z=self.reparameterize(mu,logvar)
        return z,mu,logvar




class PoseDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out
class PoseMixtureVAE(nn.Module):
    def __init__(self,
                 frame_size,
                 latent_size,
                 num_condition_frames,
                 num_future_predictions,
                 num_experts,):
        super().__init__()
        self.frame_size=frame_size
        self.latent_size=latent_size
        self.num_condition_frames=num_condition_frames
        self.num_future_predictions=num_future_predictions
        hidden_size=256
        args=(
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )
        self.encoder=PoseEncoder(*args)
        self.decoder=PoseDecoder(*args,num_experts)

    def encode(self,x,c):
        _,mu,logvar=self.encoder(x,c)
        return mu,logvar
    def forward(self,x,c):
        z,mu,logvar=self.encoder(x,c)
        return self.decoder(z,c),mu,logvar
    def sample(self,z,c,deterministic=False):
        return self.decoder(z,c)


class PoseVAE(nn.Module):
    def __init__(self,
                 frame_size,
                 latent_size,
                 num_condition_frames,
                 num_future_predictions,
                 ):
        super().__init__()
        self.frame_size=frame_size
        self.latent_size=latent_size
        self.num_condition_frames=num_condition_frames
        self.num_future_prediction=num_future_predictions

        h1=256
        self.fc1=nn.Linear(frame_size*num_condition_frames,h1)
        self.fc2=nn.Linear(frame_size+h1,h1)
        self.mu=nn.Linear(frame_size+h1,latent_size)
        self.logvar=nn.Linear(frame_size+h1,latent_size)

        self.fc4=nn.Linear(latent_size+frame_size,h1)
        self.fc5=nn.Linear(latent_size+h1,h1)
        self.out=nn.Linear(latent_size+h1,num_future_predictions)



    def encode(self,x,c):
        h1=F.elu(self.fc1(torch.cat((x,c),dim=1)))
        h2=F.elu(self.fc2(torch.cat((h1,c),dim=1)))
        s=torch.cat((x,h2),dim=1)
        return self.mu(s),self.logvar(s)
    def decode(self,z,c):
        h4=F.elu(self.fc4(torch.cat((z,c),dim=1)))
        h5=F.elu(self.fc5(torch.cat((z,h4),dim=1)))
        return self.out(torch.cat((z,h5),dim=1))

    def reparmeterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    def forward(self,x,c):
        mu,logvar=self.encode(x,c)
        z=self.reparmeterize(mu,logvar)
        return self.decode(z,c),mu,logvar

class CEncoder(nn.Module):
    def __init__(self,input,hidden,latent):
        super(CEncoder,self).__init__()
        self.fc1=nn.Linear(input,hidden)
        self.mu=nn.Linear(hidden,latent)
        self.logvar=nn.Linear(hidden,latent)
        self.elu=nn.ELU()

    def forward(self,x):
        h=self.elu(self.fc1(x))
        mu=self.mu(h)
        logvar=self.logvar(h)
        return mu,logvar


class CDecoder(nn.Module):
    def __init__(self,latent,hidden,output_numframe,output_size):
        super(CDecoder,self).__init__()
        self.fc1=nn.Linear(latent,hidden)
        self.fc2=nn.Linear(hidden,output_size*output_numframe)
        self.output_size=output_size
        self.output_numframe=output_numframe
    def forward(self,z):
        h=torch.relu(self.fc1(z))
        x_recon=torch.sigmoid(self.fc2(h))
        return x_recon.view(-1,self.output_numframe,self.output_size)

class CVAE(nn.Module):
    def __init__(self,input,hidden,latent,output_numframe,output_size):
        super(CVAE,self).__init__()
        self.encoder=CEncoder(input,hidden,latent)
        self.decoder=CDecoder(latent,hidden,output_numframe,output_size)

    def reparmeterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    def forward(self,x):
        mu,logvar=self.encoder(x)
        z=self.reparmeterize(mu,logvar)
        x_recon=self.decoder(z)
        return x_recon,mu,logvar


class PIController:
    def __init__(self,kp,ki,beta_min,beta_max):
        self.kp=kp
        self.ki=ki
        self.beta_min=beta_min
        self.beta_max=beta_max
        self.integral_error=0

    def update(self,set_point,measured_value):
        error=set_point-measured_value
        self.integral_error+=error
        beta=self.kp/(1+torch.exp(error))-self.ki*self.integral_error+self.beta_min
        beta=torch.clamp(beta,self.beta_min,self.beta_max)
        return beta



def loss_function(recon_x,x,mu,logvar,beta):
    BCE=nn.functional.binary_cross_entropy(recon_x,x,reduction='sum')
    KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+beta*KLD

def CTrain(model,data_loader,optimizer,controller,set_point,num_epoch=10):
    model.train()
    for epoch in range(num_epoch):
        for i,(data,_) in enumerate(data_loader):
            data=data.view(-1,784)
            data=Variable(data)
            optimizer.zero_grad()

            recon_batch,mu,logvar=model(data)
            KL_divergence=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())/data.size()
            beta=controller.update(set_point,KL_divergence)
            loss=loss_function(recon_batch,data,mu,logvar,beta)
            loss.backward()
            optimizer.step()

            if i%100==0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}, KL Divergence: {KL_divergence.item()}, Beta: {beta.item()}')

class GRUEncoder(nn.Module):
    def __init__(self,input,hidden,latent,num_layers,dropout=0.5):
        super(GRUEncoder,self).__init__()
        self.input=input
        self.hidden=hidden
        self.latent=latent
        self.gru=nn.GRU(input,hidden,num_layers=num_layers,dropout=dropout,batch_first=True)
        self.fc=nn.Linear(hidden*num_layers,latent)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        h0=torch.zeros(self.latent,x.size(0),self.hidden).to(x.device)
        out,hidden=self.gru(x,h0)
        out=out[:,-1,:]
        out=self.fc(out)
        out=self.dropout
