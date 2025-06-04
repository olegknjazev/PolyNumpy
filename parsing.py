from bs4 import BeautifulSoup as bs

soup = bs(html_content, 'lxml')

title = soup.find('title')
print(title)
print(type(title))
print(title.text)

print(soup.body.p)

pList = soup.body.find_all('p')
for i, p in enumerate(pList):
    print(p.text)
    print('------------------')

print([bullet.text for bullet in soup.body.find_all('li')])

p2 = soup.find(id='paragraph 2')
print(p2.text)

divClassText = soup.find_all('div')
print(divClassText)

for div in divClassText:
    id = div.get('id')
    print(id)
    print(div.text)
    print('-----------------')

soup.body.find(id='paragraph 0').decompose()
soup.body.find(id='paragraph 1').decompose()

print(soup.body.find(id='paragraphs'))

new_p = soup.new_tag('p')
print(new_p)
print(type(new_p))

new_p.string = 'Новое содержание'
print(new_p)

soup.find(id='empty').append(new_p)

print(soup)

from urllib.request import urlopen

url = 'https://yandex.ru'
html_content = urlopen(url).read()
print(html_content(:1000))

sp = bs(html_content, 'lxml')
print(sp.title.text)

