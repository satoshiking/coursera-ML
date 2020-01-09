import pandas as pd

df = pd.read_csv('titanic.csv', index_col='PassengerId')
print(df.head())
print(df.info())
print(df.shape)


#print('Unique Sex:', df['Sex'].unique())
print('Number of male on board: ', df[df['Sex'] == 'male']['Sex'].count())
print('Number of male on board: ', df[df['Sex'] == 'female']['Sex'].count())

#print('Unique survived:', df['Survived'].unique())
alive = df[df['Survived'] == 1]['Survived'].count() / df.shape[0]
print("Доля выживших пассажиров: {:.2%}".format(alive))

#print('Unique Pclass:', df['Pclass'].unique())
class1 = df[df['Pclass'] == 1]['Pclass'].count() / df.shape[0]
print("Доля пассажиров первого класса: {:.2%}".format(class1))

print("Средний возраст пассажиров: {:.2f}".format(df['Age'].mean()))
print("Медианный возраст пассажиров: {:.2f}".format(df['Age'].median()))

print(df.loc[:, ['SibSp', 'Parch']].corr())

#print(df[df['Sex'] == 'male']['Name'][15:30])



def rule(name):
    if name.find('(') != -1:
        result = name[name.find('(') + 1:].split()[0]
        result = result.replace(')', '')
        result = result.replace('"', '')
        return result
    else:
        return name[name.find('.') + 1:].split()[0]
df['first_name'] = df.apply(lambda x: rule(x['Name']), axis=1)
#Maria Marie Annie Mrs Mari Ann Anne

grouped = df[df['Sex'] == 'female'].groupby(by='first_name').count()
print(grouped.sort_values(by='Survived', ascending=False)[0:10])

print( df[ df['first_name'] == 'Mrs' ] [['Name', 'first_name']] )