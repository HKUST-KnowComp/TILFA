import torch
import torch.nn.functional as F
import torchvision.models as models
import transformers
from transformers import logging as hf_logging
from torch import nn
from torch.nn import CrossEntropyLoss, MarginRankingLoss
from transformers import AutoModelForMaskedLM, BertModel, AutoModel, AutoConfig
hf_logging.set_verbosity_error()
from torch.autograd import Variable
from transformers import BertPreTrainedModel, BertModel

from esim.layers import Seq2SeqEncoder, SoftmaxAttention
from esim.utils import replace_masked
import copy


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ImageModelResNet50(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_model=True):
        super().__init__()
        self.out_dim = out_dim
        self.resnet50 = models.resnet50(pretrained=True)
        self.avgpool = self.resnet50.avgpool
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        if freeze_model:
            for param in self.resnet50.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim))

    def forward(self, x):
        out_7x7 = self.resnet50(x).view(-1, 2048, 7, 7)
        out = self.avgpool(out_7x7).view(-1, 2048)
        # critical to normalize projections
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        # return out, attn, residual
        return out


class ImageModelResNet101(torch.nn.Module):
    def __init__(self, out_dim, freeze_model=True):
        super().__init__()
        self.out_dim = out_dim
        self.resnet101 = models.resnet101(pretrained=True)
        self.avgpool = self.resnet101.avgpool
        self.resnet101 = torch.nn.Sequential(*list(self.resnet101.children())[:-2])
        if freeze_model:
            for param in self.resnet101.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim))

    def forward(self, x):
        out_7x7 = self.resnet101(x).view(-1, 2048, 7, 7)
        out = self.avgpool(out_7x7).view(-1, 2048)
        # critical to normalize projections
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        # return out, attn, residual
        return out


class ImageModelVGG16(torch.nn.Module):
    def __init__(self, out_dim, freeze_model=True):
        super().__init__()
        self.out_dim = out_dim
        self.vgg = models.vgg16(pretrained=True)
        self.avgpool = self.vgg.avgpool
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:-1])
        if freeze_model:
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, 1024, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim, bias=True))

        # self.vgg16.classifier = self.fc

    def forward(self, x):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=-1)
        x = self.fc(x)
        return x


class TextModel(torch.nn.Module):
    def __init__(self, text_model_name, out_dim, freeze_model=True):
        super(TextModel, self).__init__()
        self.out_dim = out_dim
        # self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        self.bert_model = AutoModel.from_pretrained(text_model_name)
        if freeze_model:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        text_config = AutoConfig.from_pretrained(text_model_name)
        self.pooler = Pooler(text_config)
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_config.hidden_size, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(text_config.hidden_size, out_dim))

    def forward(self, input_ids, attention_mask):

        out = self.pooler(self.bert_model(input_ids= input_ids, attention_mask=attention_mask, return_dict=False)[0])
        out = self.fc(out)
        return out


class MultiModelResnet50(torch.nn.Module):
    def __init__(self, args, text_model_name, out_dim=1, freeze_model=True):
        super(MultiModelResnet50, self).__init__()

        self.args = args
        self.out_dim = out_dim
        # self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        self.bert_model = AutoModel.from_pretrained(text_model_name)
        if freeze_model:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.resnet50 = models.resnet50(pretrained=True)
        self.avgpool = self.resnet50.avgpool
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        if freeze_model:
            for param in self.resnet50.parameters():
                param.requires_grad = False

        self.batchnorm = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False))

        text_config = AutoConfig.from_pretrained(text_model_name)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(text_config.hidden_size + 1024, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

        if args.use_pooler == 1:
            self.pooler = Pooler(text_config)

    def forward(self, input_ids, attention_mask, image):
        if self.args.use_pooler == 1:
            text_outputs = self.pooler(self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0])
        else:
            text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0, :]

        out_7x7 = self.resnet50(image).view(-1, 2048, 7, 7)
        image_outputs = self.avgpool(out_7x7).view(-1, 2048)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out

class MultiModelResnet101(torch.nn.Module):
    def __init__(self, args, text_model_name, out_dim=1, freeze_model=True):
        super(MultiModelResnet101, self).__init__()

        self.args = args
        self.out_dim = out_dim
        # self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        self.bert_model = AutoModel.from_pretrained(text_model_name)
        if freeze_model:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.resnet101 = models.resnet101(pretrained=True)
        self.avgpool = self.resnet101.avgpool
        self.resnet101 = torch.nn.Sequential(*list(self.resnet101.children())[:-2])
        if freeze_model:
            for param in self.resnet101.parameters():
                param.requires_grad = False

        self.batchnorm = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False))

        text_config = AutoConfig.from_pretrained(text_model_name)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(text_config.hidden_size + 1024, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

        if args.use_pooler == 1:
            self.pooler = Pooler(text_config)

    def forward(self, input_ids, attention_mask, image):
        if self.args.use_pooler == 1:
            text_outputs = self.pooler(self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0])
        else:
            text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0, :]

        out_7x7 = self.resnet101(image).view(-1, 2048, 7, 7)
        image_outputs = self.avgpool(out_7x7).view(-1, 2048)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out

class MultiModelVGG16(torch.nn.Module):
    def __init__(self, args, text_model_name, out_dim=1, freeze_model=True):
        super(MultiModelVGG16, self).__init__()

        self.args = args
        self.out_dim = out_dim
        # self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        self.bert_model = AutoModel.from_pretrained(text_model_name)
        if freeze_model:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.vgg = models.vgg16(pretrained=True)
        self.avgpool = self.vgg.avgpool
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:-1])
        if freeze_model:
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, 1024, bias=True))

        text_config = AutoConfig.from_pretrained(text_model_name)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(text_config.hidden_size + 1024, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

        if args.use_pooler == 1:
            self.pooler = Pooler(text_config)

    def forward(self, input_ids, attention_mask, image):
        if self.args.use_pooler == 1:
            text_outputs = self.pooler(self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0])
        else:
            text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0, :]

        image_outputs = self.vgg(image)
        image_outputs = torch.flatten(image_outputs, 1)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out


class MultiModelLayout(torch.nn.Module):
    def __init__(self, model_args, num_labels=2, dim_common=1024, n_attn_heads=8):
        super(MultiModelLayout, self).__init__()

        if model_args.use_margin_ranking_loss == 1:
            num_labels = 1
        self.args = model_args
        self.num_labels = num_labels
        # self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        if "roberta" in model_args.text_model_name_or_path:
            self.text_model = AutoModel.from_pretrained(model_args.text_model_name_or_path)
        elif "deberta" in model_args.text_model_name_or_path:
            self.text_model = AutoModel.from_pretrained(model_args.text_model_name_or_path)
        elif "bert" in model_args.text_model_name_or_path:
            self.text_model = BertModel.from_pretrained(model_args.text_model_name_or_path)

        self.layoutlm = AutoModel.from_pretrained(model_args.model_name_or_path)

        text_config = AutoConfig.from_pretrained(model_args.text_model_name_or_path)
        layout_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

        # self.projector = torch.nn.Sequential(
        #     torch.nn.Linear(25088, 4096, bias=True),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(0.5, inplace=False),
        #     torch.nn.Linear(4096, 1024, bias=True))

        if model_args.use_pooler == 1:
            self.text_pooler = Pooler(text_config)
            self.layout_pooler = Pooler(layout_config)

        text_feature_dim = text_config.hidden_size
        visual_feature_dim = layout_config.hidden_size
        if model_args.cross_attn_type == -1:
            self.classify = torch.nn.Sequential(
                torch.nn.Linear(text_feature_dim + visual_feature_dim, 256),
                torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5, inplace=False),
                torch.nn.Linear(256, self.num_labels))
            if self.args.use_margin_ranking_loss == 1:
                self.text_classify = torch.nn.Linear(text_feature_dim, self.num_labels)

        elif model_args.cross_attn_type == 6:
            self._attention = SoftmaxAttention()
            self._projection_text = nn.Sequential(nn.Linear(4 * text_feature_dim, text_feature_dim), nn.ReLU())
            self._projection_image = nn.Sequential(nn.Linear(4 * visual_feature_dim,visual_feature_dim), nn.ReLU())
            self.classify = nn.Sequential(nn.Dropout(p=text_config.hidden_dropout_prob),  # p=dropout
                                                 nn.Linear(2 * (text_feature_dim+visual_feature_dim), round((text_feature_dim+visual_feature_dim)/2)),
                                                 nn.Tanh(),
                                                 nn.Dropout(p=text_config.hidden_dropout_prob),  # p=dropout
                                                 nn.Linear(round((text_feature_dim+visual_feature_dim)/2), num_labels))
            self.text_classify = torch.nn.Linear(text_feature_dim, self.num_labels)
        else:
            self.classify = torch.nn.Linear(text_feature_dim, self.num_labels)
            if self.args.use_margin_ranking_loss == 1:
                self.text_classify = torch.nn.Linear(text_feature_dim, self.num_labels)
            if model_args.cross_attn_type == 0:
                self._linear_1 = nn.Linear(visual_feature_dim, text_feature_dim)
                self._linear_2 = nn.Linear(text_feature_dim + visual_feature_dim, text_feature_dim)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(visual_feature_dim + text_feature_dim, visual_feature_dim)
            elif model_args.cross_attn_type == 1:
                self._linear_1 = nn.Linear(visual_feature_dim, text_feature_dim)
                self._linear_2 = nn.Linear(2 * text_feature_dim, text_feature_dim)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(2 * text_feature_dim, text_feature_dim)
            elif model_args.cross_attn_type == 2:
                self._linear_1 = nn.Linear(visual_feature_dim, text_feature_dim)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(2 * text_feature_dim, text_feature_dim)
            elif model_args.cross_attn_type == 3:
                self._linear_1 = nn.Linear(text_feature_dim, dim_common)
                self._linear_2 = nn.Linear(visual_feature_dim, dim_common)
                self._linear_3 = nn.Linear(text_feature_dim + visual_feature_dim, text_feature_dim)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(visual_feature_dim + text_feature_dim, visual_feature_dim)
            elif model_args.cross_attn_type == 4:
                self._linear_1 = nn.Linear(visual_feature_dim, dim_common)  # K
                self._linear_2 = nn.Linear(visual_feature_dim, dim_common)  # V
                self._linear_3 = nn.Linear(text_feature_dim, dim_common)  # Q
                self._multi_head_attn = nn.MultiheadAttention(dim_common, n_attn_heads)
                self._linear_4 = nn.Linear(text_feature_dim + dim_common, text_feature_dim)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(dim_common + text_feature_dim, dim_common)
            elif model_args.cross_attn_type == 5:
                dim_common = text_feature_dim
                self._linear_1 = nn.Linear(visual_feature_dim, dim_common)  # K
                self._linear_2 = nn.Linear(visual_feature_dim, dim_common)  # V
                self._linear_3 = nn.Linear(text_feature_dim, dim_common)  # Q
                self._multi_head_attn = nn.MultiheadAttention(dim_common, n_attn_heads)
                if model_args.use_forget_gate == 1:
                    self.fg = nn.Linear(dim_common + text_feature_dim, text_feature_dim)
            else:
                raise ValueError('Wrong cross_attn_type value!')


        self.dropout = text_config.hidden_dropout_prob
        self.final_layer_norm = nn.LayerNorm(text_config.hidden_size)
        self.sigmiod = nn.Sigmoid()
        if model_args.use_margin_ranking_loss == 0:
            self.loss_fct = CrossEntropyLoss()
        else:
            self.loss_fct = MarginRankingLoss(margin=model_args.margin)

    def siamese(self, a_output, b_output, a_mask, b_mask):

        attended_a, attended_b = self._attention(a_output, a_mask, b_output, b_mask)

        enhanced_a = torch.cat([a_output,
                                attended_a,
                                a_output - attended_a,
                                a_output * attended_a],
                               dim=-1)

        enhanced_b = torch.cat([b_output,
                                attended_b,
                                b_output - attended_b,
                                b_output * attended_b],
                               dim=-1)

        projected_a = self._projection_text(enhanced_a)
        projected_b = self._projection_image(enhanced_b)

        v_a_avg = torch.sum(projected_a * a_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(a_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(projected_b * b_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(b_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(projected_a, a_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(projected_b, b_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        return v

    def forward(
        self,
        text_input_ids=None,
        text_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        bbox=None,
        images=None,
    ):
        if self.args.use_margin_ranking_loss == 1:
            text_input_ids_new = copy.deepcopy(text_input_ids)
            text_attention_mask_new = copy.deepcopy(text_attention_mask)
            pooled_text_features_new = self.text_model(input_ids=text_input_ids_new, attention_mask=text_attention_mask_new, return_dict=False)[0][:, 0, :]

        text_features = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=False)[0]
        image_features= self.layoutlm(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, images=images, output_hidden_states=True)[0]
        if self.args.use_pooler == 1:
            pooled_text_features = self.text_pooler(text_features)
            pooled_image_features = self.layout_pooler(image_features)
        else:
            pooled_text_features = text_features[:, 0, :]
            pooled_image_features = image_features[:, 0, :]

        def forget_gate(image_features, text_features):  # image_features:[bs, seq_len, image_dim]
            forget_mask = self.fg(torch.cat((image_features, text_features), 2))  # [bs, seq_len, image_dim]
            forget_mask = self.sigmiod(forget_mask)
            forget_mask = F.dropout(forget_mask, p=self.dropout, training=self.training)
            image_features = forget_mask.mul(image_features)
            return image_features

        if self.args.cross_attn_type == -1:
            multi_outputs = torch.cat((pooled_text_features, pooled_image_features), 1)
            logits = self.classify(multi_outputs)

        elif self.args.cross_attn_type == 6:
            siamese_out = self.siamese(text_features, image_features, text_attention_mask, attention_mask)
            logits = self.classify(siamese_out)

        else:
            if self.args.cross_attn_type == 0:
                image_features_transformed = self._linear_1(image_features)  # [bs, image_len, text_dim=1024]
                attn = torch.bmm(text_features, image_features_transformed.transpose(1, 2))  # [bs, seq_len=512, text_dim=1024] * [bs, text_dim, image_len]
                attn = F.softmax(attn, dim=1)  # [bs, seq_len=512, image_len]
                image_features = torch.bmm(attn, image_features)  # [bs, seq_len=512, image_len]*[bs, image_len, image_dim] =  [bs, seq_len, image_dim]
                if self.args.use_forget_gate == 1:
                    image_features = forget_gate(image_features, text_features)
                output = self._linear_2(torch.cat((text_features, image_features), 2))
            elif self.args.cross_attn_type == 1:
                image_features = self._linear_1(image_features)
                attn = torch.bmm(text_features, image_features.transpose(1, 2))
                attn = F.softmax(attn, dim=1)
                image_features = torch.bmm(attn, image_features)  # (S_t, D_t)
                if self.args.use_forget_gate == 1:
                    image_features = forget_gate(image_features, text_features)
                output = self._linear_2(torch.cat((text_features, image_features), 2))
            elif self.args.cross_attn_type == 2:
                image_features = self._linear_1(image_features)
                attn = torch.bmm(text_features, image_features.transpose(1, 2))
                attn = F.softmax(attn, dim=1)
                image_features = torch.bmm(attn, image_features)  # (S_t, D_t)
                if self.args.use_forget_gate == 1:
                    image_features = forget_gate(image_features, text_features)
                output = image_features
            elif self.args.cross_attn_type == 3:
                hidden_states_transformed = self._linear_1(text_features)
                image_features_transformed = self._linear_2(image_features)
                attn = torch.bmm(hidden_states_transformed, image_features_transformed.transpose(1, 2))
                attn = F.softmax(attn, dim=1)
                image_features = torch.bmm(attn, image_features)
                if self.args.use_forget_gate == 1:
                    image_features = forget_gate(image_features, text_features)
                output = self._linear_3(torch.cat((text_features, image_features), 2))
            elif self.args.cross_attn_type == 4:
                K = self._linear_1(image_features).transpose(0, 1)
                V = self._linear_2(image_features).transpose(0, 1)
                Q = self._linear_3(text_features).transpose(0, 1)
                attn_output, _ = self._multi_head_attn(Q, K, V)
                attn_output = attn_output.transpose(0, 1)
                if self.args.use_forget_gate == 1:
                    forget_mask = self.fg(torch.cat((attn_output, text_features), 2))
                    forget_mask = self.sigmiod(forget_mask)
                    forget_mask = F.dropout(forget_mask, p=self.dropout, training=self.training)
                    attn_output = forget_mask.mul(attn_output)
                output = self._linear_4(torch.cat((text_features, attn_output), 2))
            elif self.args.cross_attn_type == 5:
                K = self._linear_1(image_features).transpose(0, 1)
                V = self._linear_2(image_features).transpose(0, 1)
                Q = self._linear_3(text_features).transpose(0, 1)
                attn_output, _ = self._multi_head_attn(Q, K, V)
                attn_output = attn_output.transpose(0, 1)
                if self.args.use_forget_gate == 1:
                    forget_mask = self.fg(torch.cat((attn_output, text_features), 2))
                    forget_mask = self.sigmiod(forget_mask)
                    forget_mask = F.dropout(forget_mask, p=self.dropout, training=self.training)
                    attn_output = forget_mask.mul(attn_output)
                output = attn_output

            output = F.dropout(output, p=self.dropout, training=self.training)
            # Residual Connection
            hidden_states = text_features + output
            if self.args.use_pooler == 1:
                hidden_states = self.text_pooler(self.final_layer_norm(hidden_states))
            else:
                hidden_states = self.final_layer_norm(hidden_states)[:, 0, :]
            logits = self.classify(hidden_states)

        loss = None
        if labels is not None:
            if self.args.use_margin_ranking_loss == 0:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                text_logits = self.text_classify(pooled_text_features_new)
                labels = (labels - 0.5) * 2
                loss = self.loss_fct(logits, text_logits, labels)
                logits = logits - text_logits

        return (loss, logits) if loss is not None else logits
