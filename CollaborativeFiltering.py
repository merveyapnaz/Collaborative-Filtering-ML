#Kütüphanelerin eklenmesi
import pandas as pd
from scipy.spatial.distance import cosine

#Verilerin okunması
data = pd.read_csv('Desktop/Collaborative_Filtering/lastfm-matrix-germany.csv')

#----Ögeye dayalı önerilerin başlangıcı----
#user sütununun atılması(Bu aşamada sadece şarkılar üzerinden işlem yapılacağı için)
data_germany = data.drop('user', 1)

#Benzerlik değerlerini hesaplamak ve saklamak adına bir Pandas DataFrame oluşturulması
data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)


#---Kosinüs benzerliğinin hesaplanması---
for i in range(0,len(data_ibs.columns)):
    for j in range(0,len(data_ibs.columns)):
        #Boşlukların kosinüs benzerlikleri ile doldurulması
        data_ibs.ix[i,j] = 1-cosine(data_germany.ix[:,i],data_germany.ix[:,j])

#Bir ögeye en yakın komşuları tutmak için yeni bir DataFrame'in oluşturulması
data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,11))

#Benzerlik DataFrame'inden Komşuların isimlerinin doldurulması
for i in range(0,len(data_ibs.columns)):
    data_neighbours.ix[i,:10] = data_ibs.ix[0:,i].sort_values(ascending=False)[:10].index
#----Ögeye dayalı önerilerin bitişi----

#----Kullanıcıya dayalı önerilerin başlangıcı----
#Benzerlik puanını hesaplayan fonksiyon
def getScore(history, similarities):
    return sum(history*similarities)/sum(similarities)

#Benzerlik verilerinin saklanması için bir DataFrame oluşturulur
#Temel olarak orjinal veri ile aynıdır ama sadece başlık kısımları doldurulur, içi boştur.
data_sims = pd.DataFrame(index=data.index,columns=data.columns)
data_sims.ix[:,:1] = data.ix[:,:1]

#Tüm satır ve sütunların benzerlik puanı ile doldurulması
for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]

        if data.ix[i][j] == 1:
            data_sims.ix[i][j] = 0

        else:
            product_top_names = data_neighbours.ix[product][1:10]
            product_top_sims = data_ibs.ix[product].sort_values(ascending=False)[1:10]
            user_purchases = data_germany.ix[user,product_top_names]

            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)
        
#Kullanıcı tabanlı öneriler matrisinin üretilmesi
#En iyi şarkıların alınması
data_recommend = pd.DataFrame(index=data_sims.index, columns=['user','1','2','3','4','5','6'])
data_recommend.ix[0:,0] = data_sims.ix[:,0]

#Matrisin benzerlik puanları yerine şarkı isimleri ile doldurulması
for i in range(0,len(data_sims.index)):
    data_recommend.ix[i,1:] = data_sims.ix[i,:].sort_values(ascending=False).ix[1:7,].index.transpose()

print(data_recommend.ix[:10,:4])