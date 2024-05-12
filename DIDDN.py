import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertPreTrainedModel, BertConfig
from torch.distributions.normal import Normal


class BERTVAEModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTVAEModel, self).__init__(config)
        self.bert = BertModel(config)
        self.cls_layer1 = nn.Linear(config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128, 128)
        self.tanh1 = nn.Tanh()
        self.ff_Z = nn.Linear(128, 1)
        self.fc_mu_theta = nn.Linear(128, 1)
        self.fc_sigma_theta = nn.Linear(128, 1)
        self.fc_mu_b = nn.Linear(128, 1)
        self.fc_sigma_b = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        x = self.cls_layer1(logits)
        x = self.relu1(x)
        x = self.ff1(x)
        x = self.tanh1(x)
        z = self.ff_Z(x)
        mu_theta = self.fc_mu_theta(x)
        sigma_theta = torch.exp(self.fc_sigma_theta(x))
        mu_b = self.fc_mu_b(x)
        sigma_b = torch.exp(self.fc_sigma_b(x))
        return z, mu_theta, sigma_theta, mu_b, sigma_b


class DIDDN(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(DIDDN, self).__init__()
        self.beta = 0.1
        bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert_vae_model = BERTVAEModel(bert_config)
        self.states = {}

    def initialize_author(self, author_id):
        if author_id not in self.states:
            self.states[author_id] = {
                'Z1': torch.zeros(1, 1),
                'Z2': torch.zeros(1, 1),
                'theta': torch.zeros(1, 1),
                'b': torch.zeros(1, 1),
                'Y_hat': torch.zeros(1, 1),
                'last_Y_hat': torch.zeros(1, 1),
                'last_theta': torch.zeros(1, 1),
                'last_b': torch.zeros(1, 1)
            }

    def exponential_moving_average(self, old, new, update=True):
        return new * self.beta + old * (1 - self.beta) if update else old

    def forward(self, author_id, topic_label, input_ids, attention_mask):
        if author_id not in self.states:
            self.initialize_author(author_id)

        z, mu_theta, sigma_theta, mu_b, sigma_b = self.bert_vae_model(input_ids, attention_mask)
        state = self.states[author_id]
        z1, z2, theta, b = state['Z1'], state['Z2'], state['theta'], state['b']

        update_z1 = topic_label == 0
        update_z2 = topic_label == 1

        z1 = self.exponential_moving_average(z1, z, update=update_z1)
        z2 = self.exponential_moving_average(z2, z, update=update_z2)

        if update_z1 or update_z2:
            theta = self.exponential_moving_average(theta, Normal(mu_theta, sigma_theta).sample())
            b = self.exponential_moving_average(b, Normal(mu_b, sigma_b).sample())

        weights = torch.cat((0.5 + theta, 0.5 - theta), dim=1)
        y_hat = torch.cat((z1, z2), dim=1) @ weights.T + b

        self.states[author_id].update({
            'Z1': z1,
            'Z2': z2,
            'theta': theta,
            'b': b,
            'Y_hat': y_hat,
            'last_theta': theta.detach(),
            'last_b': b.detach()
        })

        return y_hat


def loss_function(predictions, targets, mu_theta, sigma_theta, mu_b, sigma_b,
                  prior_theta_variance, prior_b_variance, theta_last, b_last,
                  theta_current, b_current, epoch, n_epochs, epsilon=1e-8):

    mse_loss = nn.MSELoss()(predictions, targets)
    kl_loss_theta = 0.5 * (torch.log(prior_theta_variance + epsilon) - torch.log(sigma_theta.pow(2) + epsilon) +
                           (sigma_theta.pow(2) + mu_theta.pow(2)) / (prior_theta_variance + epsilon) - 1)
    kl_loss_b = 0.5 * (torch.log(prior_b_variance + epsilon) - torch.log(sigma_b.pow(2) + epsilon) +
                       (sigma_b.pow(2) + mu_b.pow(2)) / (prior_b_variance + epsilon) - 1)
    stability_weight = (epoch / n_epochs) ** 2
    stability_loss_theta = stability_weight * torch.abs(theta_current - theta_last).mean()
    stability_loss_b = stability_weight * torch.abs(b_current - b_last).mean()
    total_loss = mse_loss + kl_loss_theta + kl_loss_b + stability_loss_theta + stability_loss_b
    return total_loss


def train(model, data_loader, n_epochs, learning_rate, device):
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    prior_theta_variance = torch.tensor(0.15, device=device)
    prior_b_variance = torch.tensor(0.8, device=device)

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch in data_loader:
            input_ids, attention_mask, targets, author_ids, topic_labels = [x.to(device) for x in batch]

            optimizer.zero_grad()

            batch_loss = 0

            for input_id, mask, target, author_id, topic_label in zip(input_ids, attention_mask, targets, author_ids,
                                                                      topic_labels):
                y_hat = model(author_id, topic_label, input_id, mask)

                state = model.states[author_id.item()]
                theta_current, b_current = state['theta'], state['b']
                theta_last, b_last = state['last_theta'], state['last_b']

                loss = loss_function(y_hat, target.unsqueeze(0), state['mu_theta'], state['sigma_theta'], state['mu_b'],
                                     state['sigma_b'],
                                     prior_theta_variance, prior_b_variance,
                                     theta_last, b_last, theta_current, b_current,
                                     epoch, n_epochs)

                loss.backward()

                batch_loss += loss.item()

            optimizer.step()

            epoch_loss += batch_loss / len(input_ids)

        loss_history.append(epoch_loss / len(data_loader))

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(data_loader)}")

    return loss_history


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:

            input_ids, attention_mask, author_ids, topic_labels = [x.to(device) for x in batch]

            batch_predictions = []
            for input_id, mask, author_id, topic_label in zip(input_ids, attention_mask, author_ids, topic_labels):
                y_hat = model(author_id, topic_label, input_id, mask)
                batch_predictions.append(y_hat.cpu().numpy())

            predictions.extend(batch_predictions)
    return predictions
