class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(128)
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(128)
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.bn1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.bn2(output)
        output = self.ff2(output)
        return output


def train(model, criterion1, criterion2, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion2(output, target.type_as(output))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training loss is {train_loss / len(train_loader)}")
        val_loss = evaluate(model=model, criterion1=criterion1, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss : {}".format(epoch, val_loss))


def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)

            predicted_label += output
            actual_label += target

    return predicted_label
