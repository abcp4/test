from ftplib import FTP_TLS # FTP with TLS (Transport Layer Security)

def inverse_list(alist):
    return dict(zip(alist,range(len(alist))))

def rotate_list(l, n):
    return l[n:] + l[:n]

def pcut(alist, ppart):
    partitions = []
    last = 0
    for pi in ppart:
        cut = int(len(alist)*pi)
        partitions.append(alist[last:last+cut])
        last += cut
    return partitions

def send_to_ftp(filepath):
    ftps = FTP_TLS('ftpes.learn.inf.puc-rio.br') # Define Ftp server
    ftps.login('learn', 'LRepoAdm18!!') # Login to ftp server
    ftps.prot_p() # Enable data encryption
    ftps.retrlines('LIST') # List Directory
    ftps.cwd('/projeto/01_tarefas_de_geofisica/QualiSismo/CV_results') # Change Directory
    ftps.storlines("STOR " + filepath, open(filepath,'rb'))
    