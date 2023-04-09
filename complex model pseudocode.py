#complex model
def complex_model():
    X= tokenizer(p)  
    for id in author_ids:       
        hidden_state_1_output=bert(X)   
        Z= multiply(Z,topic.T)+multiply(hidden_state_1_output,topic.T)*(1-beta) +multiply(Z,topic.T)*beta 
    
        hidden_state_2_output=linear(theta)
        hidden_state_2_output=clamp(hidden_state_2_output,-0.5,0.5)
        theta=hidden_state_2_output*(1-beta)+theta*beta
    
        hidden_state_3_output=linear(b)
        hidden_state_3_output=clamp(hidden_state_3_output,0,1)
        b= hidden_state_3_output*(1-beta)+b*beta

        Y=Y=Z*[0.5+theta,0.5-theta].T+b

        return Y, Z, theta, b

def train():
    for epoch in n_epochs:
        for (post,id,t) in dataloader:
            Y_hat, Z, theta, b =complex_model(post,id,t)

        # for y in Y:
            miu_z_given_y = miu_z_given_y*beta + (1-beta)*sum(Z)/n
            sigma_z_given_y = sigma_z_given_y*beta + (1-beta)*(sum((Z-miu_y)**2)/n-1)  #moving average

        for id in author_ids:
            for y in Y_hat:
                sigma_z = tensor.append(Z.std()).mean()
 
        sigma_y_hat=y_hat.var()
        simga_z_given_y_hat = get_variance(sigma_y_hat,sigma_z,theta)
        miu_z_given_y_hat = get_mean(Z,y,theta,b)

        loss=nn.MSELoss(miu_z_given_y,miu_z_given_y_hat)+MSELoss(sigma_z_given_y,simga_z_given_y_hat)  #MSE(miu&sigma)

        optimizer.zero_grad()
        loss.backward
        optimizer.step()