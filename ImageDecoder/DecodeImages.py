
decoder = Decoder()

device = torch.device('cuda')

dis_A_state_dict = torch.load("model_decoder-99.0")

# obtain the state dictionary of a previously trained model

decoder.load_state_dict(dis_A_state_dict, strict = False)

# send dictionary to device

decoder.to(device)


# ***********************************************************************************

toxicPeptides = np.load("super_toxicNumpy.npy")

peptideSequences = []
for peptide in toxicPeptides:
      
    peptideSequence = ""
    for i in range(peptide.shape[2]):
        aminoAcid = peptide[:,:,i]
        aminoAcid = np.expand_dims(aminoAcid, axis=0)
#       
        aminoAcidProbabilities = decoder(Variable(torch.cuda.FloatTensor(aminoAcid)), 0)
        aminoAcidProbabilities = as_np(aminoAcidProbabilities)
      
        aminoAcidLetter = getAminoAcid(aminoAcidProbabilities[0])
        peptideSequence = peptideSequence + aminoAcidLetter
        
    peptideSequences.append(peptideSequence)
#     print(peptideSequence)

peptideSequences = list(set(peptideSequences))
print(len(peptideSequences))
    
import pickle
superToxicFile = open("superToxicSequences.pickle", "wb")
pickle.dump(peptideSequences, superToxicFile)

# ***********************************************************************************

antiPeptides = np.load("super_antiNumpy.npy")

peptideSequences = []
for peptide in antiPeptides:
      
    peptideSequence = ""
    for i in range(peptide.shape[2]):
        aminoAcid = peptide[:,:,i]
        aminoAcid = np.expand_dims(aminoAcid, axis=0)
#       
        aminoAcidProbabilities = decoder(Variable(torch.cuda.FloatTensor(aminoAcid)), 0)
        aminoAcidProbabilities = as_np(aminoAcidProbabilities)
      
        aminoAcidLetter = getAminoAcid(aminoAcidProbabilities[0])
        peptideSequence = peptideSequence + aminoAcidLetter
        
    peptideSequences.append(peptideSequence)
#     print(peptideSequence)

peptideSequences = list(set(peptideSequences))
print(len(peptideSequences))
    
import pickle
superAntiFile = open("superAntiSequences.pickle", "wb")
pickle.dump(peptideSequences, superAntiFile)


