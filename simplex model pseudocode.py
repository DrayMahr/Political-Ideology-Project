#simplex model
def simplex_model():
    X= tokenizer(p)   #posts
    for id in author_ids:       #each author has only one Z,theta and b
        hidden_state_1_output=bert(X)   #using pretrained-Bert-model as regressor
        Z= multiply(hidden_state_1_output,topic.T)*(1-beta) +multiply(Z,topic.T)*beta  # moving average  (1-beta)*Zt+beta*Zt-1
    
        hidden_state_2_output=linear(theta)
        hidden_state_2_output=clamp(hidden_state_2_output,-0.5,0.5)
        theta=hidden_state_2_output*(1-beta)+theta*beta
    
        hidden_state_3_output=linear(b)
        hidden_state_3_output=clamp(hidden_state_3_output,0,1)
        b= hidden_state_3_output*(1-beta)+b*beta


        Y=Z*[0.5+theta,0.5-theta].T+b

    return Y, Z, theta, b


def train():
    for epoch in n_epochs:
        for (post,id,t) in dataloader:
            Y_hat, Z, theta, b =simplex_model(post,id,t)

        loss=(Y-Y_hat)**2 + lambd1 * l2_penalty(square(theta)) + lambd2 * l2_penalty(b)  #加一个stable constraint

        optimizer.zero_grad()
        loss.backward
        optimizer.step()