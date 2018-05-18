from io_data import read_data, write_data
import numpy as np


def Kmeans(obs_data,mus,epsilon):
    """ returns matrix r: expressing which observed data point belongs to which cluster
                vectors mus: mean vector of each cluster"""
    N = len(obs_data)
    K = len(mus)
    tmp = np.ones(mus.shape)
    i=1

    def Astep(obs_data,mus):
        """Assignment step"""
        r = np.zeros((N,K))
        min_ix = np.argmin(np.array([[np.linalg.norm(x-mu) for mu in mus] for x in obs_data]),axis = 1)
        r[np.arange(N),min_ix] = 1
        return r

    def Ustep(obs_data,r):
        """Update step"""
        Nk= np.sum(r,axis=0)
        Nk[Nk==0]=1
        mus = np.array([np.sum([r[n, k] * obs_data[n] for n in range(N)], axis=0) / Nk[k] for k in range(K)])
        return mus

    while (np.abs((mus - tmp))>epsilon).all():
        tmp = mus.copy()
        r = Astep(obs_data,mus)
        mus = Ustep(obs_data,r)
        print("K-Means Iteration: {}".format(i))
        i+=1
    print("K-Means Converged")


    return mus,r

def EM(obs_data,pis,mus,sigmas,epsilon):
    """ returns matrix gamma: expressing the probability observed data point belongs to a given cluster
                vectors mus: mean vector of each cluster
                matrices sigmas: covariance matrix of each cluster
                vector pis: mixing coefficients"""
    N = len(obs_data)
    K = len(pis)
    tmp = 1
    i = 1

    def MultiNorm(x, mu, sigma):
        """ Multivariate normal distribution density"""
        return np.linalg.det(2 * np.pi * sigma) ** (-1 / 2) * np.exp(
            -((x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu)) / 2)

    def LogMultiNorm(x, mu, sigma):
        """ Analytical form of the log of the multivariate normal distribution density"""
        return -(np.log(np.linalg.det(2 * np.pi * sigma)) + (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu)) / 2

    def ExpLogLikelihood(obs_data,pis,mus,sigmas,gamma):
        """ Computes the expected log-likelihood (Q in the report) """
        return np.sum(np.array(
            [np.sum(np.array(
                [gamma[n,k]*(np.log(pis[k])+LogMultiNorm(obs_data[n],mus[k],sigmas[k]))
                 for k in range(K)]
            )) for n in range(N)]))

    def Estep(obs_data,pis,mus,sigmas):
        """Expectation step"""
        normal = pis*np.array([[MultiNorm(x,mu,sigma) for  x in obs_data] for mu,sigma in zip(mus,sigmas)]).T.reshape(N,-1)
        norm = np.sum(normal,axis=1).reshape(N,-1)
        norm[norm == 0] = 1
        return normal/norm

    def Mstep(obs_data,gamma):
        """Maximization step"""
        Nk= np.sum(gamma,axis=0)
        pis = Nk/N
        mus = np.array([np.sum([gamma[n,k]*obs_data[n] for n in range(N)],axis=0)/Nk[k] for k in range(K)])
        sigmas = np.array([np.sum([gamma[n,k]*(obs_data[n]-mus[k]).dot((obs_data[n]-mus[k]).T) for n in range(N)],axis=0)/Nk[k] for k in range(K)])
        return pis,mus,sigmas

    gamma = Estep(obs_data, pis, mus, sigmas)
    pis, mus, sigmas = Mstep(obs_data, gamma)

    while np.abs((ExpLogLikelihood(obs_data,pis,mus,sigmas,gamma)-tmp)/tmp)>epsilon:
        tmp = ExpLogLikelihood(obs_data,pis,mus,sigmas,gamma)
        print(tmp)
        gamma = Estep(obs_data,pis,mus,sigmas)
        pis,mus,sigmas = Mstep(obs_data,gamma)
        print("EM iteration: {}".format(i))
        i+=1

    print("EM Converged")

    return gamma,pis,mus,sigmas



def BuildMask(data,gamma):
    mask = np.full(data.shape,255)
    mask[:,:2] = data[:,:2]
    min_ix = np.argmin(gamma, axis=1)
    mask[np.arange(len(data)),2+ min_ix] = 0
    return(mask[:,:3])

def FBgnd(data,mask):
    """returns the foreground and the background of the image using the previously built mask"""
    d = data.shape[-1]-2
    Fgnd = data.copy()
    Bgnd = data.copy()
    for n in range(len(data)):
        Fgnd[n,2:] = np.zeros(d) if mask[n,-1]==255 else Fgnd[n,2:]
        Bgnd[n, 2:] = np.zeros(d) if mask[n, -1] == 0 else Bgnd[n, 2:]
    return Fgnd,Bgnd



#Importing data and initialization
animal = "owl"
K = 2
K_epsilon = 1e-5
EM_epsilon = 1e-2
data,image = read_data("../a2/"+animal+".txt",False)

N = len(data)
d = data.shape[-1] - 2
obs_data = data[:,2:].reshape(N,d,1)

mus = np.random.randn(K,d,1)
sigmas = np.tile(np.identity(d),(K,1)).reshape(K,d,d)
pis = np.repeat(1/K,K)


#Using K-Means to provide the initial means for EM
mus,r = Kmeans(obs_data,mus,K_epsilon)

#Using EM for data segmentation (expressed in gamma)
gamma,pis,mus,sigmas = EM(obs_data,pis,mus,sigmas,EM_epsilon)
print(pis)

#Generating mask, foreground, background and exporting them
mask = BuildMask(data,gamma)
Fgnd,Bgnd = FBgnd(data,mask)

write_data(mask,"../outputs/a2/"+ animal +"_mask.txt")
read_data("../outputs/a2/"+animal + "_mask.txt", True, save=True)

write_data(Fgnd,"../outputs/a2/"+animal +"_Fgnd.txt")
read_data("../outputs/a2/"+animal + "_Fgnd.txt", False, save = True)

write_data(Bgnd,"../outputs/a2/"+animal +"_Bgnd.txt")
read_data("../outputs/a2/"+animal + "_Bgnd.txt", False, save = True)







