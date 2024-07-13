
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



phones=pd.read_csv("C:\\PhonesAnalysis\\SamsungPhonesPrices.csv")
print(phones)

#Eliminarea rândurilor care conțin valori lipsă
phones = phones.dropna()

#Eliminarea coloanelor care conțin valori lipsă
phones = phones.dropna(axis=1)


#Lista de telefoane

phoneList=['Galaxy S4', 'Galaxy S5', 'Galaxy S II', 'Galaxy A50', 'Galaxy A52S', 'Galaxy A21'] #hardcodat
phoneModels=phones.iloc[:20,2].tolist()
#afisare lista
print('Lista e: ', phoneModels)
#lungime lista
print('Lungimea listei e: ', len(phoneModels))
#adaugare element la sfarsitul listei si afisare
phoneModels.append('Galaxy S22')
print('Lista e: ', phoneModels)
#stergere element din lista
phoneModels.remove('Galaxy S22')
print('Lista e: ', phoneModels)
#inserare element in lista la o pozitie
phoneModels.insert(3, 'Galaxy S II')
print('Lista e: ', phoneModels)
#stergerea ultimului element din lista
phoneModels.pop()
print('Lista e: ', phoneModels)
#inversare lista
phoneModels.reverse()
print("Lista (invers) e: ", phoneModels)
print('\n')

#Set de telefoane

phoneSet={'Galaxy S4', 'Galaxy S5', 'Galaxy S II', 'Galaxy A50', 'Galaxy A52S', 'Galaxy A21'} #hardcodat
phoneModelsSet=set(phoneModels)
#afisare set
print('Setul e: ', phoneModelsSet)
#lungime set
print('Lungimea setului e: ', len(phoneModelsSet))
#adaugare element la sfarsitul setului si afisare
phoneModelsSet.add('Galaxy S22')
print('Setul e: ', phoneModelsSet)
#stergere element din set
phoneModelsSet.remove('Galaxy S II')
print('Setul e: ', phoneModelsSet)
#adaugare element deja existent
phoneModelsSet.add('Galaxy S22')
print('Setul e: ', phoneModelsSet)
#stergerea unui element din set random
phoneModelsSet.pop()
print('Setul e: ', phoneModelsSet)
#adaugarea unui alt set in set
phoneModelsSet.update('Galaxy Z Fold4','Galaxy A14 5G', 'Galaxy S23 Ultra')
print("Setul e: ", phoneModelsSet)
print('\n')

#Tuplu de telefoane

#impachetare
phoneTuple=('Galaxy S4', 'Galaxy S5', 'Galaxy S II', 'Galaxy A50', 'Galaxy A52S', 'Galaxy A21') #hardcodat
phoneModelsTuple = tuple(phoneModels)
#despachetare
(primul,al_doilea,al_treilea,al_patrulea,al_cincilea,al_saselea,al_saptelea,al_optulea,al_noualea,al_zecelea,*restul,antepenultimul,penultimul,ultimul) = phoneModelsTuple
#afisare tuplu
print('Tuplul e: ', phoneModelsTuple)
#lungime tuplu
print('Lungimea tuplului e: ', len(phoneModelsTuple))
#adaugare element la sfarsitul tuplului si afisare
phoneModelsTuple = list(phoneModelsTuple)
phoneModelsTuple.append('Galaxy S22')
phoneModelsTuple = tuple(phoneModelsTuple)
print('Tuplul e: ', phoneModelsTuple)
#stergere element din tuplu
phoneModelsTuple = list(phoneModelsTuple)
phoneModelsTuple.remove('Galaxy S II')
phoneModelsTuple = tuple(phoneModelsTuple)
print('Tuplul e: ', phoneModelsTuple)
#inserare element in tuplu la o pozitie
phoneModelsTuple = list(phoneModelsTuple)
phoneModelsTuple.insert(3, 'Galaxy S II')
phoneModelsTuple = tuple(phoneModelsTuple)
print('Tuplul e: ', phoneModelsTuple)
print(al_treilea)
#stergerea ultimului element din tuplu
phoneModelsTuple = list(phoneModelsTuple)
phoneModelsTuple.pop()
phoneModelsTuple = tuple(phoneModelsTuple)
print('Tuplul e: ', phoneModelsTuple)
#inversare tuplu
phoneModelsTuple = list(phoneModelsTuple)
phoneModelsTuple.reverse()
phoneModelsTuple = tuple(phoneModelsTuple)
print("Tuplul (invers) e: ", phoneModelsTuple)
print('\n')

#Dictionar de telefoane

phoneDict={'Galaxy S4':1500, 'Galaxy S20':4000, 'Galaxy A52':1800, 'Galaxy S7':1600, 'Galaxy Note II':3300} #hardcodat
phonePrices=phones.set_index('model').to_dict()['starting_price_dollars']
#afisare dictionar
print('Dictionar: ', phonePrices)
#afisarea pretului telefonului Galaxy S4
print('Pret Galaxy S4 : ', phonePrices.get('Galaxy S4'))
#afisarea telefoane si preturi
print('fiecare telefon din dictionar si pretul sau corespondent: ', phonePrices.items())
#afisarea telefoanelor din lista
print('Telefoane: ', phonePrices.keys())
#stergerea telefonului Galaxy S4
phonePrices.pop('Galaxy S4')
print('Telefoane(fara Galaxy S4): ', phonePrices)
#afisarea preturilor telefoanelor
print('Preturile: ', phonePrices.values())
print('\n')



#structuri conditionate si structuri repetitive (se mai adauga cateva telefoane in lista PhoneModels)

phoneModels.append('Galaxy J2 Core')
phoneModels.append('Galaxy A52')
phoneModels.append('Galaxy A53')
#afisarea tuturor telefoanelor de forma Galaxy A___
print('Telefoane Galaxy A: ')
listPhones=[]
for phone in phoneModels:
    if(len(phone)>9):
        if(phone[7]=='A'):
            listPhones.append(phone)
print(listPhones)

#Functii

phonesList=['Galaxy S4', 'Galaxy S5', 'Galaxy S II', 'Galaxy A50', 'Galaxy A52S', 'Galaxy A21']
priceList=[50,60,80,45,90,67]
#parcurgerea listei si afisarea pretului
print('Afisare preturi: ')
def afisare_preturi(telefoane,preturi):
    for i in range(len(telefoane)):
        telefon='Telefonul '+telefoane[i]+' are pretul '+str(preturi[i])
        print(telefon)

afisare_preturi(phonesList,priceList)
print('\n')

#marirea preturilor dintr-o lista care sunt sub 70 de dolari
def marire_preturi(telefoane,preturi):
    for i in range(len(telefoane)):
        if preturi[i]<70:
            preturi[i]=preturi[i]*2.5
marire_preturi(phoneList,priceList)
print('Marire preturi: ',priceList)
print('\n')

#modificare date pachet pandas
secondPhones=phones.copy()
print(secondPhones.iloc[:,1:7])

#marim pretul cu 100 de dolari pt telefoanele care sunt fabricate in anul 2018  si au pretul sub 300 de dolari
secondPhones.loc[(phones['starting_price_dollars']<300)&(phones['year']==2018),'starting_price_dollars']+=100
print('Modificare preturi pachet pandas ',secondPhones.iloc[:,1:7])
print('\n')

#afisare coloane si inregistrari cu loc si iloc
print('apelare iloc primele 3 coloane si primele 3 linii')
print(secondPhones.iloc[[0,1,2],0:3])
print('apelare loc de la coloana manufacturer la year')
print(secondPhones.loc[0:3,'manufacturer':'year'])
print('\n')

#Merge seturi de date

preturi = pd.read_csv('C:\\PhonesAnalysis\\SetPreturi.csv')
telefoane = pd.read_csv('C:\\PhonesAnalysis\\SetTelefoane.csv')

result = pd.merge(telefoane[['id','model']],
                 preturi[['id','starting_price_dollars']],
                  on='id',
                  how='outer',
                  indicator=True)
print('Merge result:')
print(result)


#Prelucrari statistice

print('Prelucrari statistice ')
print('\n')
#afisare pret minim,maxim si total de telefoane
print('Pret minim ',phones['starting_price_dollars'].min())
print('Pret maxim ',phones['starting_price_dollars'].max())
print('Totalul de telefoane',phones['model'].count())
print('\n')
#utilizare groupby
#form- touchscreen, slider, bar si flip phone
group_data=phones.groupby('form')['starting_price_dollars']
print('Pretul maxim in functie de forma ',group_data.max())
print('\n')
#utilizare agreare
print('Agregare suma preturi conform formei ',secondPhones.groupby(['form']).agg({'starting_price_dollars':sum}))
print('\n')

#Grafic reprezentand totalul pentru fiecare tip al telefoanelor (Pie)

plot_data1=group_data.count()
ax = plot_data1.sort_values().plot(kind='pie', autopct='%.3f%%',colors=['red','cyan','purple','pink'],labels=plot_data1.index)
ax.set_title("Procentul telefoanelor de fiecare tip")
plt.ylabel("")
plt.show()
print('Datele din grafic ',plot_data1)

#Grafic pentru forma telefonului si media pretului (Bar)

plot_data2=group_data.mean()
ax=plot_data2.sort_values().plot(kind="bar",color='pink')
ax.set_title('Media de pret pentru fiecare tip de telefon')
plt.show()
print('Datele din grafic ',plot_data2)

#Stergere coloane si inregistrari

phones = phones.drop(['manufacturer', 'year'], axis=1)
phones = phones.drop([0, 1, 2], axis=0)
print(phones)

#Salvare telefoane cu preturi marite in format csv

secondPhones.to_csv('C:\\PhonesAnalysis\\date_formatateTelefoane.csv')

#Regresie logistica

#separarea atributelor si variabilei smartphone
y = phones[['smartphone']]
x = phones['units_sold_m']

y= y.replace({'Yes': 1, 'No': 0})
y = y.values.ravel()
#impartirea setului de date in train si test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
# modelul de regresie logistica
model = LogisticRegression()
# antrenarea modelului pe setul de train
model.fit(X_train, y_train)
# realizarea de predictii pe setul de test
y_pred = model.predict(X_test)
print('\n')
print("Predictii regresie logistica: ")
print(y_pred)

#Regresie Multipla
X = phones[['smartphone', 'units_sold_m']]
y = phones['starting_price_dollars']
X_copy = X.loc[:, 'smartphone'].copy()
X_copy = X_copy.replace({'Yes': 1, 'No': 0})
X.loc[:, 'smartphone'] = X_copy
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print('\n')
print(results.params)
print('\n')
print(results.pvalues)