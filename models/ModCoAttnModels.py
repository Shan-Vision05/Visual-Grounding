import torch
import torch.nn as nn
from TextModels import *
from VisionModels import *
from transformers import BertTokenizer

class CoAttentionScorer(nn.Module):
    def __init__(self, feature_dim=512, nheads=2):
        super().__init__()

        self.CrossAttention_Image = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=nheads,
            dropout=0.1,
            batch_first=True
        )

        self.CrossAttention_Loc = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=nheads,
            dropout=0.1,
            batch_first=True
        )

        # self.CrossAttention_Fuse = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=nheads,
        #     dropout=0.0,
        #     batch_first=True
        # )

        # self.SelfAttention_Img = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=nheads,
        #     dropout=0.0,
        #     batch_first=True
        # )

        # self.SelfAttention_Loc = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=nheads,
        #     dropout=0.0,
        #     batch_first=True
        # )


        self.Norm_Image = nn.LayerNorm(feature_dim)
        self.Norm_Loc = nn.LayerNorm(feature_dim)

        self.Norm_Image_1 = nn.LayerNorm(feature_dim)
        self.Norm_Loc_1 = nn.LayerNorm(feature_dim)

        self.Norm_Fuse = nn.LayerNorm(feature_dim)

        # self.ClassificationHead = nn.Sequential(
        #     nn.Linear(10*feature_dim, 5*feature_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(5*feature_dim),
        #     nn.Linear(5*feature_dim, feature_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(feature_dim),
        #     # nn.Dropout(0.3),
        #     nn.Linear(feature_dim, 10)
        # )

        self.ClassificationHead = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),

            # nn.Dropout(0.),
            nn.Linear(feature_dim, feature_dim*2),
            nn.GELU(),
            nn.LayerNorm(feature_dim*2),

            # nn.Dropout(0.2),
            nn.Linear(feature_dim*2, feature_dim//4),
            nn.GELU(),
            nn.LayerNorm(feature_dim//4),

            nn.Dropout(0.4),
            nn.Linear(feature_dim//4, feature_dim//8),
            nn.GELU(),
            nn.LayerNorm(feature_dim//8),

            # nn.Dropout(0.2),
            nn.Linear(feature_dim//8, 1)
        )

        self.RawImgWeight = nn.Parameter(torch.tensor(0.5))
        self.RawLocweight = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_img_features, text_loc_features, img_features, loc_features):

        # print(text_img_features.shape)
        # print(img_features.shape)
        # self.RawImgWeight = self.RawImgWeight.to(loc_features.device)
        # self.RawLocweight = self.RawLocweight.to(loc_features.device)

        object_features = img_features.mean(dim=[3,4])
        text_img_features = text_img_features.unsqueeze(dim=1)
        text_loc_features = text_loc_features.unsqueeze(dim=1)


        attn_img_out, _ = self.CrossAttention_Image(
            query=object_features,
            key=text_img_features,
            value=text_img_features
        )
        # print(attn_img_out.shape)
        # print(text_img_features.shape)
        # print(object_features.shape)

        attn_img_out = self.Norm_Image(attn_img_out + object_features)

        attn_loc_out, _ = self.CrossAttention_Loc(
            query=loc_features,
            key=text_loc_features,
            value=text_loc_features
        )
        attn_loc_out = self.Norm_Loc(attn_loc_out + loc_features)

        # print(attn_img_out.shape)
        # print(attn_loc_out.shape)


        weights = torch.softmax(
            torch.stack([self.RawImgWeight, self.RawLocweight]), dim=0
        )

        attn_img_out = weights[0] * attn_img_out
        attn_loc_out = weights[1] * attn_loc_out

        # attn_img_out_2, _ = self.SelfAttention_Img(
        #     query= attn_img_out,
        #     key= attn_img_out,
        #     value= attn_img_out
        # )

        # attn_img_out_2 = self.Norm_Image_1(attn_img_out + attn_img_out_2)



        # attn_loc_out_2, _ = self.SelfAttention_Loc(
        #     query= attn_loc_out,
        #     key= attn_loc_out,
        #     value= attn_loc_out
        # )

        # attn_loc_out_2 = self.Norm_Loc_1(attn_loc_out + attn_loc_out_2)


        # print(attn_img_out_2.shape)
        # print(attn_loc_out_2.shape)

        # attn_fuse_out, _ = self.CrossAttention_Fuse(
        #     query=attn_img_out,
        #     key=attn_loc_out,
        #     value=attn_loc_out
        # )
        # attn_fuse_out = self.Norm_Fuse(attn_fuse_out + attn_img_out)

        attn_fuse_out = torch.cat((attn_img_out, attn_loc_out), dim=-1)

        # print(attn_fuse_out.shape)
        # x = attn_fuse_out.squeeze(1)
        # print('x')
        # print(x.shape)

        scores = self.ClassificationHead(attn_fuse_out)
        scores = scores.squeeze(-1)
        return scores
    
class VisualGrounding(nn.Module):
    def __init__(self, device):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = tokenizer.vocab_size

        self.visionEncoder = VisionEncoder()
        self.subjectTextEncoder = SubjectTextEncoder(vocab_size=vocab_size)
        self.coAttentionScorer = CoAttentionScorer()

        self.locTextEncoder = LocationTextEncoder(vocab_size=vocab_size)
        self.locVisionEncoder = LocationVisionEncoder(device)

    def forward(self, image, input_ids, attn_mask):

        final_result, img_features = self.visionEncoder(image)
        text_img_features = self.subjectTextEncoder(input_ids.squeeze(dim=1), attn_mask.squeeze(dim=1))
        text_loc_features = self.locTextEncoder(input_ids.squeeze(dim=1), attn_mask.squeeze(dim=1))
        loc_features = self.locVisionEncoder(final_result)
        # print(img_features)

        # scores = self.coAttentionScorer(text_features, img_features)
        scores = self.coAttentionScorer(text_img_features, text_loc_features, img_features, loc_features)

        return final_result, scores