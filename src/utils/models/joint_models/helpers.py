#Wrappers
class JoinedModels(nn.Module):
    def __init__(self, vision_model, classifier):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier

    def forward(self, x):
        features = self.vision_model(x)  # image -> features
        logits = self.classifier(features)  # features -> prediction
        return logits
class TiledJoinedModels(nn.Module):

    def __init__(self, vision_model, classifier):
        # Fixed the mismatch: super() should match the current class name
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier

    def forward(self, x):
        # x shape: (B, n, C, H, W)
        B, n, C, H, W = x.size()

        # 1. Collapse Batch and Tile dimensions into a single batch
        # New shape: (B * n, C, H, W)
        x_flattened = x.view(B * n, C, H, W)

        # 2. Forward pass through vision model in parallel (1 call instead of n)
        # Expected output shape per image: (B * n, feature_dim)
        features_flattened = self.vision_model(x_flattened)

        # 3. Reshape back to separate Batch and Tiles, then flatten tiles into features
        # Shape transition: (B * n, feature_dim) -> (B, n * feature_dim)
        feature_dim = features_flattened.size(-1)
        combined_feats = features_flattened.view(B, n * feature_dim)

        # 4. Final classification
        return self.classifier(combined_feats)
class ConcatenateViews(nn.Module):
    def __init__(self, vision_model):
        # Fixed the mismatch: super() should match the current class name
        super().__init__()
        self.vision_model = vision_model

    def forward(self, x):
        # x shape: (B, n, C, H, W)
        B, n, C, H, W = x.size()

        # 1. Collapse Batch and Tile dimensions into a single batch
        # New shape: (B * n, C, H, W)
        x_flattened = x.view(B * n, C, H, W)

        # 2. Forward pass through vision model in parallel (1 call instead of n)
        # Expected output shape per image: (B * n, feature_dim)
        features_flattened = self.vision_model(x_flattened)

        # 3. Reshape back to separate Batch and Tiles, then flatten tiles into features
        # Shape transition: (B * n, feature_dim) -> (B, n * feature_dim)
        feature_dim = features_flattened.size(-1)
        features = features_flattened.view(B, n, feature_dim)  # ← Changed this line

        
        return features

#Helpers time-sequences
def _property_encoder(m, d, dropout):
    """Small MLP: raw property vector -> d-dim embedding.
    Last layer is zero-initialised, so at init the model is identical to the property-free one."""
    enc = nn.Sequential(nn.Linear(m, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
    nn.init.zeros_(enc[-1].weight)
    nn.init.zeros_(enc[-1].bias)
    return enc
