import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, truncnorm, norm
import warnings
warnings.filterwarnings("ignore")

# ----------------------- FUNCTIONS -----------------------

def gibbs_sampler(mu_1, sigma_1, mu_2, sigma_2, sigma_ts, samples):
    # Gibbs Sampling
    if samples < 100:
        print('Too few samples. Samples must be larger than 100\n')
        return

    t = np.zeros(samples)
    s_1 = np.zeros(samples)
    s_2 = np.zeros(samples)

    # Initial values
    s_1[0] = mu_1
    s_2[0] = mu_2
    t[0] = 0

    # Intervals for truncnorm
    myclip_a = 0
    myclip_b = 1000
    my_mean = mu_1 - mu_2
    my_std = sigma_ts
    
    # Matrix
    A = np.array([[1, -1]])
    mu = np.array([[mu_1], [mu_2]])
    sigma_matrix1 = np.array([[sigma_1**2, 0], [0, sigma_2**2]])
    sigma_matrix2 = np.linalg.inv(np.linalg.inv(sigma_matrix1)+np.transpose(A)*(1/sigma_ts**2)@A)

    for i in range(samples-1):
        mu_ts = sigma_matrix2@(np.linalg.inv(sigma_matrix1)@mu+np.transpose(A)*(t[i]/sigma_ts**2))
        mu_ts = np.ravel(mu_ts)
        s_1[i+1], s_2[i+1] = np.random.multivariate_normal(mu_ts, sigma_matrix2)
        my_mean = s_1[i+1] - s_2[i+1]
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        t[i+1] = truncnorm.rvs(a, b, my_mean, my_std)

    # Removing burn-in
    s_1 = s_1[200:]
    s_2 = s_2[200:]

    return s_1, s_2

def gibbs_sampler_burnin(mu_1, sigma_1, mu_2, sigma_2, sigma_ts, samples):
    # Gibbs Sampling
    if samples < 100:
        print('Too few samples. Samples must be larger than 100\n')
        return

    t = np.zeros(samples)
    s_1 = np.zeros(samples)
    s_2 = np.zeros(samples)

    s_1[0] = mu_1
    s_2[0] = mu_2
    t[0] = 0

    s_1mean = np.empty(samples)
    s_2mean = np.empty(samples)
    s_1mean[0] = mu_1
    s_2mean[0] = mu_2
    
    # Intervals for truncnorm
    myclip_a = 0
    myclip_b = 1000
    my_mean = mu_1 - mu_2
    my_std = sigma_ts
    
    # Matrix
    A = np.array([[1, -1]])
    mu = np.array([[mu_1], [mu_2]])
    sigma_matrix1 = np.array([[sigma_1**2, 0], [0, sigma_2**2]])
    sigma_matrix2 = np.linalg.inv(np.linalg.inv(sigma_matrix1)+np.transpose(A)*(1/sigma_ts**2)@A)

    for i in range(samples-1):
        mu_ts = sigma_matrix2@(np.linalg.inv(sigma_matrix1)@mu+np.transpose(A)*(t[i]/sigma_ts**2))
        mu_ts = np.ravel(mu_ts)
        s_1[i+1], s_2[i+1] = np.random.multivariate_normal(mu_ts, sigma_matrix2)
        my_mean = s_1[i+1] - s_2[i+1]
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        t[i+1] = truncnorm.rvs(a, b, my_mean, my_std)
        s_1mean[i+1] = np.sum(s_1)/(i+2)
        s_2mean[i+1] = np.sum(s_2)/(i+2)
        
    return s_1mean, s_2mean

def output_values_1(s_1, s_2):
    # Initial values
    # mu = 1
    # sigma = 1
    mu = 1 
    sigma = 1
    #x = np.linspace(mu-5*sigma,mu+5*sigma,100)

    # Calculate mean and standard deviation
    mu_s1 = np.mean(s_1)
    std_s1 = np.std(s_1)
    S1 = norm.pdf(x, mu_s1, std_s1)
    mu_s2 = np.mean(s_2)
    std_s2 = np.std(s_2)
    S2 = norm.pdf(x, mu_s2, std_s2)
    return mu_s1, std_s1, S1, mu_s2, std_s2, S2

def output_values_2(s_1, s_2):
    # Initial values
    # mu = 1
    # sigma = 1
    mu = 25 
    sigma = 25/3
    #x = np.linspace(mu-5*sigma,mu+5*sigma,100)

    # Calculate mean and standard deviation
    mu_s1 = np.mean(s_1)
    std_s1 = np.std(s_1)
    S1 = norm.pdf(x, mu_s1, std_s1)
    mu_s2 = np.mean(s_2)
    std_s2 = np.std(s_2)
    S2 = norm.pdf(x, mu_s2, std_s2)
    return mu_s1, std_s1, S1, mu_s2, std_s2, S2

def predict(mu_1, sigma_1, mu_2, sigma_2, samples):
    # Generate samples
    s_1 = np.random.normal(mu_1, sigma_1, samples)
    s_2 = np.random.normal(mu_2, sigma_2, samples)

    # Check score
    t = np.mean(s_1 - s_2)
    y = np.sign(t)
    return(y)

def prediction_accuracy(prediction, data):
    results = []
    count = 0
    score = data['score1'] - data['score2']
    # Calculating y for the actual scores
    for i in range(data.shape[0]):
        if score[i] > 0:
            results.append(1)
        elif score[i] < 0:
            results.append(-1)
        else:
            results.append(0)

    # Comparing with our predictions
    for k in range(len(results)):
        if results[k] == prediction[k]:
            count += 1

    draws = results.count(0)
    accuracy = count/len(results)
    accuracy_no_draws = count/(len(results)-draws)
    # Printing out results
    print("Number of correct predictions:", count)
    print("Prediction accuracy: "+ str(round((accuracy*100),2)) + "%")
    print("Prediction accuracy if draws are disregarded:" + str(round((accuracy_no_draws*100),2)) + "%")
    
def multiplyGauss(mu_1, sigma_1, mu_2, sigma_2):
    # Given function from session 9
    sigma = 1/(1/sigma_1+1/sigma_2)
    mu = (mu_1/sigma_1+mu_2/sigma_2)*sigma
    return mu, sigma 

def divideGauss(mu_1, sigma_1, mu_2, sigma_2):
    # Given function from session 9
    mu, sigma = multiplyGauss(mu_1, sigma_1, mu_2, -sigma_2)
    return mu, sigma   

def truncGaussMM(my_a, my_b, mu_1, sigma_1):
    # Given function from session 9
    a, b = (my_a - mu_1) / np.sqrt(sigma_1), (my_b - mu_1) / np.sqrt(sigma_1)
    mu = truncnorm.mean(a, b, loc=mu_1, scale=np.sqrt(sigma_1))
    sigma = truncnorm.var(a, b, loc=0, scale=np.sqrt(sigma_1))
    return mu, sigma
# ----------------------- Q5 part 1 -----------------------
mu_1 = 1
mu_2 = 1
sigma_1 = 1
sigma_2 = 1 
sigma_ts = 1 
samples = 500

s_1, s_2 = gibbs_sampler_burnin(mu_1, sigma_1, mu_2, sigma_2, sigma_ts, samples)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(s_1,label="s1", color='red')
plt.legend(loc='upper right')
plt.title('Mean of s1')
plt.subplot(1, 2, 2)
plt.plot(s_2,label="s2", color='blue')
plt.legend(loc='upper right')
plt.title('Mean of s2')
plt.show()

# ----------------------- Q5 part 2 -----------------------
#Parameters
x = np.linspace(mu_1-5*sigma_1,mu_1+5*sigma_1,100)

times = []
samples_vec = [500, 1000, 1500, 2000]
time_start = time.process_time()
for i in range(len(samples_vec)):
    s_1, s_2 = gibbs_sampler(mu_1, sigma_1, mu_2, sigma_2, sigma_ts, samples_vec[i])
    mu_s1, std_s1, S1, mu_s2, std_s2, S2 = output_values_1(s_1,s_2)
    time_elapsed = (time.process_time() - time_start)
    times.append(time_elapsed)

    # ----------------------- Q5 part 3 -----------------------
    #Make a plot
    plt.figure(i+1)
    plt.hist(s_1, label="s1", bins =50, color='red', alpha = 0.3, density=True)
    plt.plot(x, S1, linewidth=2, color='red', label="Approx. s1")
    plt.hist(s_2, label="s2", bins =50, color='blue', alpha = 0.3, density=True)
    plt.plot(x, S2, linewidth=2, color='blue', label="Approx. s2")
    plt.legend(loc='upper right')
    plt.title('Number of samples: ' + str(samples_vec[i]))
    plt.show()

#print(times)

# ----------------------- Q5 part 4 -----------------------
S1_prior = norm.pdf(x, mu_1, sigma_1)
S2_prior = norm.pdf(x, mu_2, sigma_2)
plt.figure(6)
plt.subplot(1, 2, 1)
plt.plot(x, S1, linewidth=2, color='red', label="Posterior s1")
plt.plot(x, S1_prior, linewidth=2, color='blue', label="Prior s1")
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.plot(x, S2, linewidth=2, color='red', label="Posterior s2")
plt.plot(x, S2_prior, linewidth=2, color='blue', label="Prior s2")
plt.legend(loc='upper right')
plt.show()

# ----------------------- Q6 -----------------------
mu = 25
sigma = 25/3
sigma_ts = 25/3
samples = 1000
x = np.linspace(mu-5*(sigma**2),mu+5*(sigma**2),100)

data = pd.read_csv("SerieA.csv") 
players = data['team1'].drop_duplicates()
data = data.sample(frac=1).reset_index(drop=True)
playerlist = dict() 

#start values for mean and variance for each team
for t in players:
    playerlist[t] = (mu,sigma) 

for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    if score > 0:
        winner = data.iloc[i]['team1']
        loser = data.iloc[i]['team2']
    else:
        winner = data.iloc[i]['team2']
        loser = data.iloc[i]['team1']
    s_1, s_2 =  gibbs_sampler(playerlist[winner][0],playerlist[winner][1],playerlist[loser][0],playerlist[loser][1],sigma_ts,samples)
    mu_1, sigma_1, S1, mu_2, sigma_2, S2 = output_values_2(s_1,s_2)
    playerlist[winner] = [mu_1, sigma_1]
    playerlist[loser] = [mu_2, sigma_2]

df = pd.DataFrame.from_dict(playerlist, orient='index', columns=['Skill', 'Standard Deviation'])
df.sort_values(by='Skill', inplace=True, ascending = False)
df['Rank'] = range(1, len(df) + 1)
print(df)

#----------------------- Q7 -----------------------
mu = 25
sigma = 25/3
samples = 300

data = pd.read_csv("SerieA.csv") 
players = data['team1'].drop_duplicates()
#data = data.sample(frac=1).reset_index(drop=True) # uncomment for un-ordered list
playerlist = {}
predictions = []

# Set initial mean and variance
for player in players:
   playerlist[player] = (mu, sigma)

# Predict matches outcome based on previous data
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    predictions.append(predict(playerlist[data.iloc[i]['team1']][0], playerlist[data.iloc[i]['team1']][1],playerlist[data.iloc[i]['team2']][0], playerlist[data.iloc[i]['team2']][1],samples))
    if score > 0:
        winner = data.iloc[i]['team1']
        loser = data.iloc[i]['team2']
        s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], playerlist[loser][1],sigma,samples)
        mu_1, sigma_1, S1, mu_2, sigma_2, S2 = output_values_2(s_1,s_2)
        playerlist[winner] = [mu_1, sigma_1]
        playerlist[loser] = [mu_2, sigma_2]
    
    elif score < 0:
        winner = data.iloc[i]['team2']
        loser = data.iloc[i]['team1']
        s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1], playerlist[loser][0], playerlist[loser][1],sigma,samples)
        mu_1, sigma_1, S1, mu_2, sigma_2, S2 = output_values_2(s_1,s_2)
        playerlist[winner] = [mu_1, sigma_1]
        playerlist[loser] = [mu_2, sigma_2]

# Calculate and print out statistics
prediction_accuracy(predictions, data) 

# ----------------------- Q9 -----------------------

mu = 25
sigma = 25/3
sigma_ts = 25/3
y0 = 1

# Message from factor f_s1 to node s1
mu_3 = mu
sigma_3 = sigma**2

# Message from factor f_s2 to node s2
mu_4 = mu
sigma_4 = sigma**2

# Message from node s1 to factor f_st
mu_5 = mu_3
sigma_5 = sigma_3

# Message from node s2 to factor f_st
mu_6 = mu_4
sigma_6 = sigma_4

# Message from factor f_st to node t
mu_7 = mu_3 - mu_4
sigma_7 = sigma_3 + sigma_4 + sigma_ts**2

# Do moment matching of the marginal of t
if y0==1:
    a, b = 0, 1000
else:
    a, b = -1000, 0

#Turning the truncated Gaussian into a Gaussian
pt_mu, pt_sigma = truncGaussMM(a,b,mu_7,sigma_7)

#Compute the updated message from f_yt to t
mu_2, sigma_2 = divideGauss(pt_mu,pt_sigma,mu_7,sigma_7)

# Compute the message from node t to factor f_st
mu_7_back = mu_2
sigma_7_back = sigma_2 + sigma_ts**2

# Compute the message from factor f_st to node s1
mu_5_back = mu_5 + mu_7_back # winner
sigma_5_back = sigma_7_back + sigma_5

# Compute the message from factor f_st to node s2
mu_6_back = mu_6 - mu_7_back # loser
sigma_6_back = sigma_7_back + sigma_6

# Compute the marginal of s1 and s2
ps1_mu, ps1_sigma = multiplyGauss(mu_3, sigma_3, mu_5_back, sigma_5_back)
ps2_mu, ps2_sigma = multiplyGauss(mu_4, sigma_4, mu_6_back, sigma_6_back)

# Figure
samples = 1000
#x = np.linspace(ps1_mu-ps1_sigma, ps2_mu + ps2_sigma, samples)
x = np.linspace(mu-5*(sigma**2),mu+5*(sigma**2),samples)

#Draw values from the Gaussian distributions
s1_norm = np.random.normal(ps1_mu, np.sqrt(ps1_sigma), samples)
s2_norm = np.random.normal(ps2_mu, np.sqrt(ps2_sigma), samples)

mu_s1 = np.mean(s1_norm)
mu_s2 = np.mean(s2_norm)
sigma_s1 = np.var(s1_norm)
sigma_s2 = np.var(s2_norm)

#Making pdf:s
s1_pdf = norm.pdf(x, mu_s1, np.sqrt(sigma_s1))
s2_pdf = norm.pdf(x, mu_s2, np.sqrt(sigma_s2))

S1, S2 = gibbs_sampler(mu, sigma, mu, sigma, sigma, samples)
mu_1_gibbs, sigma_1_gibbs, s1, mu_2_gibbs, sigma_2_gibbs, s2 = output_values_2(S1,S2) 

S1_prior = norm.pdf(x, mu_1_gibbs, sigma_1_gibbs)
S2_prior = norm.pdf(x, mu_2_gibbs, sigma_2_gibbs)

#Make a plot
plt.figure(7)
plt.hist(S1, label="Winner, GS", bins =50, color='red', alpha = 0.3, density=True)
plt.plot(x, s1_pdf, linewidth=2, color='red', label="Winner, MP")
plt.plot(x, S1_prior, linewidth=2, color='maroon', label="Winner, GS")
plt.hist(S2, label="Looser, GS", bins =50, color='blue', alpha = 0.3, density=True)
plt.plot(x, s2_pdf, linewidth=2, color='blue', label="Looser, MP")
plt.plot(x, S2_prior, linewidth=2, color='navy', label="Looser, GS")
plt.legend(loc='upper right')
plt.title('Message passing vs Gibbs sampling')
plt.xlim([-20, 70])
plt.show()

# ----------------------- Q10 -----------------------
mu = 25
sigma = 25/3
samples = 300
x = np.linspace(mu-5*sigma,mu+5*sigma,100)

# Football data
data =pd.read_csv("SerieA.csv")
players = data['team1'].drop_duplicates()
data = data.sample(frac=1).reset_index(drop=True) # uncomment for un-ordered list
playerlist = {}
predictions = []

# Set initial mean and variance
for player in players:
    playerlist[player] = (mu, sigma)

# Predict matches outcome based on previous data
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    predictions.append(predict(playerlist[data.iloc[i]['team1']][0], playerlist[data.iloc[i]['team1']][1],playerlist[data.iloc[i]['team2']][0], playerlist[data.iloc[i]['team2']][1],samples))
    if score > 0:
        winner = data.iloc[i]['team1']
        loser = data.iloc[i]['team2']
        if score == 2:
            s_1, s_2 = gibbs_sampler(1.1*playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], (1/1.1)*playerlist[loser][1],sigma,samples)
        elif score >= 3:
            s_1, s_2 = gibbs_sampler(1.2*playerlist[winner][0], playerlist[winner][1],(1/1.2)*playerlist[loser][0], playerlist[loser][1],sigma,samples)
        else:
            s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], playerlist[loser][1],sigma,samples)     
    else:
        winner = data.iloc[i]['team2']
        loser = data.iloc[i]['team1']
        if score == -2:
            s_1, s_2 = gibbs_sampler(1.1*playerlist[winner][0], playerlist[winner][1],(1/1.1)*playerlist[loser][0], playerlist[loser][1],sigma,samples)
        elif score <= -3:
            s_1, s_2 = gibbs_sampler(1.2*playerlist[winner][0], playerlist[winner][1],(1/1.2)*playerlist[loser][0], playerlist[loser][1],sigma,samples)   
        else:
            s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], playerlist[loser][1],sigma,samples)
    mu_1, sigma_1, S1, mu_2, sigma_2, S2 = output_values_2(s_1,s_2)
    playerlist[winner] = [mu_1, sigma_1]
    playerlist[loser] = [mu_2, sigma_2]

# Calculate and print out statistics
print('For the football data:')
prediction_accuracy(predictions, data) 

# Hockey (SHL) data
col_list = ["Datum","team1", "team2", "score1", "score2"]
data = pd.read_csv("SHL.csv", usecols=col_list)
players = data['team1'].drop_duplicates()
data = data.sample(frac=1).reset_index(drop=True) # uncomment for un-ordered list
playerlist = {}
predictions = []

# Set initial mean and variance
for player in players:
    playerlist[player] = (mu, sigma)

# Predict matches outcome based on previous data
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    predictions.append(predict(playerlist[data.iloc[i]['team1']][0], playerlist[data.iloc[i]['team1']][1],playerlist[data.iloc[i]['team2']][0], playerlist[data.iloc[i]['team2']][1],samples))
    if score > 0:
        winner = data.iloc[i]['team1']
        loser = data.iloc[i]['team2']
        if score == 2:
            s_1, s_2 = gibbs_sampler(1.1*playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], (1/1.1)*playerlist[loser][1],sigma,samples)
        elif score >= 3:
            s_1, s_2 = gibbs_sampler(1.2*playerlist[winner][0], playerlist[winner][1],(1/1.2)*playerlist[loser][0], playerlist[loser][1],sigma,samples)
        else:
            s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], playerlist[loser][1],sigma,samples)     
    else:
        winner = data.iloc[i]['team2']
        loser = data.iloc[i]['team1']
        if score == -2:
            s_1, s_2 = gibbs_sampler(1.1*playerlist[winner][0], playerlist[winner][1],(1/1.1)*playerlist[loser][0], playerlist[loser][1],sigma,samples)
        elif score <= -3:
            s_1, s_2 = gibbs_sampler(1.2*playerlist[winner][0], playerlist[winner][1],(1/1.2)*playerlist[loser][0], playerlist[loser][1],sigma,samples)   
        else:
            s_1, s_2 = gibbs_sampler(playerlist[winner][0], playerlist[winner][1],playerlist[loser][0], playerlist[loser][1],sigma,samples)
    mu_1, sigma_1, S1, mu_2, sigma_2, S2 = output_values_2(s_1,s_2)
    playerlist[winner] = [mu_1, sigma_1]
    playerlist[loser] = [mu_2, sigma_2]

# Calculate and print out statistics
print('For the hockey data:')
prediction_accuracy(predictions, data) 