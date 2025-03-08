import torch

class embedding_transformer(torch.nn.Module):
    def __init__(self, embed_dim = 768, identity=False):
        super(embedding_transformer, self).__init__()
        self.transformer = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        if identity:
            "Got identity inititalization"
            self.transformer.weight.data = torch.eye(embed_dim, dtype=torch.float32)
    def forward(self, x):
        x = self.transformer(x)
        return x

class EmbeddingOrthogonalizer(torch.nn.Module):
    def __init__(self, embed_dim=768, num_bases=10):
        super(EmbeddingOrthogonalizer, self).__init__()
        # Trainable bases: shape (num_bases, embed_dim)
        self.bases = torch.nn.Parameter(torch.randn(num_bases, embed_dim))

    def forward(self, x):
        # Compute projection matrix to orthogonalize the space
        V = self.bases.T  # Shape: (embed_dim, num_bases)
        V = V.to(torch.float32)  # Ensure the dtype is float32 for stability
        VtV = V.T @ V  # Shape: (num_bases, num_bases)
        VtV_inv = torch.inverse(VtV)  # Perform inversion in float32
        P = torch.eye(V.shape[0], device=V.device, dtype=torch.float32) - V @ VtV_inv @ V.T  # Projection matrix

        # Orthogonalize the input embeddings
        x = x @ P  # Orthogonalize the embeddings
        return x
    

def get_transformer(args):  
    if args.CLIP_model == 'ViT-L/14@336px':
        embed_dim = 768
    elif args.CLIP_model == 'ViT-B/16':
        embed_dim = 512
    elif args.CLIP_model == 'RN50':
        embed_dim = 1024 

    if args.num_bases == 0:
        if args.init_weight == 'i':
            print("Got identity initialization")
            transformer = embedding_transformer(embed_dim=embed_dim, identity=True)
        else:
            transformer = embedding_transformer(embed_dim=embed_dim)
    else:
        transformer = EmbeddingOrthogonalizer(embed_dim=embed_dim, num_bases=args.num_bases)
        
    transformer.to(args.device)
    return transformer
