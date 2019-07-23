
# ***********************************************************
# Load the trained discriminator

discriminator = Discriminator()

device = torch.device('cuda')

dis_A_state_dict = torch.load("transfer2_model_dis-35.0")

# obtain the state dictionary of a previously trained model

discriminator.load_state_dict(dis_A_state_dict, strict = False)

# send dictionary to device

discriminator.to(device)

# ************************************************************

rPeptides = np.load("randomPeptideImages.npy")

print(rPeptides.shape)

randomProbabilities = []
i = 0

for peptide in rPeptides:
#     print(peptide)
    peptide = np.expand_dims(peptide, axis=0)
    peptide = Variable(torch.FloatTensor(peptide))
    peptide = peptide.cuda()
    probabilities = discriminator(peptide, 0)
    i = i + 1
    randomProbabilities.append(as_np(probabilities.cpu())[0][1])
    
print(np.mean(randomProbabilities))
print(np.std(randomProbabilities))

# ************************************************************

antiPeptides = np.load("antiDomain.npy")

print(antiPeptides.shape)

antiProbabilities = []
i = 0

for peptide in antiPeptides:
#     print(peptide)
    peptide = np.expand_dims(peptide, axis=0)
    peptide = Variable(torch.FloatTensor(peptide))
    peptide = peptide.cuda()
    probabilities = discriminator(peptide, 0)
    i = i + 1
    antiProbabilities.append(as_np(probabilities.cpu())[0][1])
    
print(np.mean(antiProbabilities))
print(np.std(antiProbabilities))

# ************************************************************

toxicPeptides = np.load("toxicDomain.npy")

print(toxicPeptides.shape)

toxicProbabilities = []
i = 0

for peptide in toxicPeptides:
#     print(peptide)
    peptide = np.expand_dims(peptide, axis=0)
    peptide = Variable(torch.FloatTensor(peptide))
    peptide = peptide.cuda()
    probabilities = discriminator(peptide, 0)
    i = i + 1
    toxicProbabilities.append(as_np(probabilities.cpu())[0][1])
    
print(np.mean(toxicProbabilities))
print(np.std(toxicProbabilities))

# ************************************************************

saPeptides = np.load("super_antiNumpy.npy")

print(saPeptides.shape)

saProbabilities = []
i = 0

for peptide in saPeptides:
#     print(peptide)
    peptide = np.expand_dims(peptide, axis=0)
    peptide = Variable(torch.FloatTensor(peptide))
    peptide = peptide.cuda()
    probabilities = discriminator(peptide, 0)
    i = i + 1
    saProbabilities.append(as_np(probabilities.cpu())[0][1])
    
print(np.mean(saProbabilities))
print(np.std(saProbabilities))

# **************************************************************

stPeptides = np.load("super_toxicNumpy.npy")

print(stPeptides.shape)

stProbabilities = []
i = 0

for peptide in stPeptides:
#     print(peptide)
    peptide = np.expand_dims(peptide, axis=0)
    peptide = Variable(torch.FloatTensor(peptide))
    peptide = peptide.cuda()
    probabilities = discriminator(peptide, 0)
    i = i + 1
    stProbabilities.append(as_np(probabilities.cpu())[0][1])
    
print(np.mean(stProbabilities))
print(np.std(stProbabilities))

# **************************************************************


import seaborn as sns
import pandas as pd

print(len(stProbabilities))
print(len(saProbabilities))
print(len(randomProbabilities))
print(len(toxicProbabilities))
print(len(antiProbabilities))


dictionary = {
    "superToxicProbabilities":stProbabilities,
    "superAntiProbabilities":saProbabilities,
    "randomProbabilities":randomProbabilities,
    "toxicProbabilities":toxicProbabilities,
    "antiProbabilities":antiProbabilities[1:] # This one had 1 extra peptide
    
}


df = pd.DataFrame(dictionary)



# **********************************************************************

s = sns.distplot(df["superToxicProbabilities"].dropna(),color="red")
s = sns.distplot(df["superAntiProbabilities"].dropna(),color="blue")
s = sns.distplot(df["randomProbabilities"].dropna(), color="green")
s = sns.distplot(df["toxicProbabilities"].dropna(),color="yellow")
s = sns.distplot(df["antiProbabilities"].dropna(), color="purple")


fig = s.get_figure()
# If using google colabs:
# from google.colab import files
# fig.savefig("Dist.png")
# files.download("Dist.png") 

