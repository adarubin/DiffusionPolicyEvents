# Same trained architecture; full strict EMA load replaces all initialization.
RESNET_FEATURE_DIM = 512
PROPRIO_DIM = 2 * CFG['state_per_observation']
ACTION_DIM = CFG['action_dim']
UNET_DIMS = CFG['unet_dims']
TIME_EMBED_DIM = CFG['time_embed_dim']
UNET_KERNEL = CFG['unet_kernel']
GROUP_NORM_GROUPS = CFG['group_norm_groups']

def freeze_resnet_except_layer4(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in backbone.layer4.parameters():
        parameter.requires_grad = True
    return backbone

class ResNet18Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        freeze_resnet_except_layer4(backbone)
        self.backbone = backbone
        self.output_dim = RESNET_FEATURE_DIM
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, x):
        features = self.backbone(x)
        return features.flatten(1)

class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        embedding = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

class FiLMResBlock1D(nn.Module):

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        return h + self.skip(x)

class ConditionalUNet1D(nn.Module):

    def __init__(self, action_dim, cond_dim=256):
        super().__init__()
        c1, c2, c3 = UNET_DIMS
        self.time_embed = nn.Sequential(SinusoidalEmbedding(TIME_EMBED_DIM), nn.Linear(TIME_EMBED_DIM, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input_proj = nn.Conv1d(action_dim, c1, 1)
        self.down1 = FiLMResBlock1D(c1, c1, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.downsample1 = nn.Conv1d(c1, c2, 3, stride=2, padding=1)
        self.down2 = FiLMResBlock1D(c2, c2, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.downsample2 = nn.Conv1d(c2, c3, 3, stride=2, padding=1)
        self.down3 = FiLMResBlock1D(c3, c3, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.mid = FiLMResBlock1D(c3, c3, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.up2 = nn.Conv1d(c3 + c2, c2, 3, padding=1)
        self.up2_block = FiLMResBlock1D(c2, c2, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.up1 = nn.Conv1d(c2 + c1, c1, 3, padding=1)
        self.up1_block = FiLMResBlock1D(c1, c1, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.out_norm = nn.GroupNorm(GROUP_NORM_GROUPS, c1)
        self.out = nn.Conv1d(c1, action_dim, 1)

    def forward(self, noisy_actions, condition, timesteps):
        cond = condition + self.time_embed(timesteps)
        x = self.input_proj(noisy_actions.transpose(1, 2))
        skip1 = self.down1(x, cond)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, cond)
        x = self.downsample2(skip2)
        x = self.down3(x, cond)
        x = self.mid(x, cond)
        x = F.interpolate(x, size=skip2.shape[-1], mode='nearest')
        x = self.up2(torch.cat([x, skip2], dim=1))
        x = self.up2_block(x, cond)
        x = F.interpolate(x, size=skip1.shape[-1], mode='nearest')
        x = self.up1(torch.cat([x, skip1], dim=1))
        x = self.up1_block(x, cond)
        x = F.silu(self.out_norm(x))
        x = self.out(x)
        return x.transpose(1, 2)

class VisualConditionedDiffusionPolicy(nn.Module):

    def __init__(self):
        super().__init__()
        self.agentview_encoder = ResNet18Encoder()
        self.wrist_encoder = ResNet18Encoder()
        fusion_in = 4 * RESNET_FEATURE_DIM + PROPRIO_DIM
        self.condition = nn.Sequential(nn.Linear(fusion_in, 1024), nn.SiLU(), nn.Linear(1024, 256))
        self.unet = ConditionalUNet1D(ACTION_DIM, cond_dim=256)

    def encode_observation(self, batch):
        agent = batch['agentview']
        wrist = batch['robot0_eye_in_hand']
        batch_size = agent.shape[0]
        agent_flat = agent.reshape(batch_size * 2, 3, agent.shape[-2], agent.shape[-1])
        wrist_flat = wrist.reshape(batch_size * 2, 3, wrist.shape[-2], wrist.shape[-1])
        agent_features = self.agentview_encoder(agent_flat).reshape(batch_size, 2, -1)
        wrist_features = self.wrist_encoder(wrist_flat).reshape(batch_size, 2, -1)
        fused = torch.cat([agent_features[:, 0], agent_features[:, 1], wrist_features[:, 0], wrist_features[:, 1], batch['proprio']], dim=1)
        return self.condition(fused)

    def predict_noise(self, batch, noisy_actions, timesteps):
        condition = self.encode_observation(batch)
        return self.unet(noisy_actions, condition, timesteps)
